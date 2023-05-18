"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
valfile = "\n"
num_samples = 10 # number of samples to draw
max_new_tokens = 386 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt0
directory = 'data/risk/test/' # Replace with the path to your directory
file = open("example.txt", "w")
for filename in os.listdir(directory):
    if filename.endswith('.txt'):

        with open(directory + filename) as f:
            data = f.read()

    data = data.split('\n')
    x = 'p'
    n = 1
    while x == 'p':
        x = data[n][0]
        n = n+1
    n = n - 1
    past = data[:int(n)]
    past = '\n'.join(past)
    past = past + '\n' + "future"
    print(past)
    gt = data[int(n):]
    gtarray = []
    for i in range(1, len(gt) - 1):
          line = gt[i].strip().split('\t')
          line = [float(j) for j in line[1:]]
          gtarray.append(line)
    gtarray = np.asarray(gtarray)
    prediction = gtarray
    start_ids = encode(past)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                for j in range(len(prediction)):
                    y = model.generate(x, 1, temperature=temperature, top_k=top_k)
                    x = torch.cat((x, y), dim=1)
                    o = decode(y[0].tolist())
                    file.write(o)
                    if o == '\n':
                        continue
                    line = o;
                    while o != '\n':
                        y = model.generate(x, 1, temperature=temperature, top_k=top_k)
                        x = torch.cat((x, y), dim=1)
                        o = decode(y[0].tolist())
                        file.write(o)
                        line = line + o;
                        #print(o)
                        #print(decode(y[0].tolist()))
                    line = line.strip().split('\t')
                    print(line)
                    for i in range(1, 4):
                        if len(line) > i:
                            prediction[j][i-1] = line[i]
                        else:
                            prediction[j][i-1] = 0
    gtarray = gtarray.T[1:]
    gtarrayx = gtarray[0]
    gtarrayy = gtarray[1]
    gtarrayz = gtarray[2]
    prediction = prediction.T[1:]
    predictionx = prediction[0]
    predictiony = prediction[1]
    predictionz = prediction[2]
    dx = gtarrayx - predictionx
    dy = gtarrayy - predictiony
    dz = gtarrayz - predictionz
    p = []
    for i in range(len(dx)):
        p.append(np.sqrt(dx[i]**2 + dy[i]**2+ dz[i]**2))
        print('ade point%i:  ' + str(p[i]), i)
    print(np.mean(p))

                    # with open("output.txt", "w") as file:
                    #     print(decode(y[0].tolist()), file = file)
                    # # with open("output.txt", 'r') as f:
                    # #     data = []
                    # #     for line in f:
                    # #         c
                    # #         line = [float(i) for i in line]
                    # #         data.append(line)
                    # #     data = np.asarray(data)
                    # with open("gt.txt", 'r') as f:
                    #     data_gt = []
                    #     for line in f:
                    #         line = line.strip().split('\t')
                    #         line = [float(i) for i in line]
                    #         data_gt.append(line)
                    #     data_gt = np.asarray(data_gt)
                    #     data_gt = data_gt.T[2:]
                        # print(pred_traj_gtp)
                    # fig = plt.figure(figsize=(8, 6))
                    # ax = fig.add_subplot(projection='3d')
                    # ax.set_title('NanoGPT predict vertical landing')
                    # ax.set_xlabel('x')
                    # ax.set_ylabel('y')
                    # ax.set_zlabel('z')
                    # obs_trajp = data.T[2:]
                    # obs_trajpx = obs_trajp[0]
                    # obs_trajpy = obs_trajp[1]
                    # obs_trajpz = obs_trajp[2]
                    # gen_trajx = obs_trajpx[10:20];
                    # gen_trajy = obs_trajpy[10:20];
                    # gen_trajz = obs_trajpz[10:20];
                    # print(gen_trajz)
                    # data_gtx = data_gt[0]
                    # data_gty = data_gt[1]
                    # data_gtz = data_gt[2]
                    # print(data_gtz)
                    # dx = data_gtx - gen_trajx
                    # dy = data_gty - gen_trajy
                    # dz = data_gtz - gen_trajz
                    # p = []
                    # print(dz)
                    # for i in range(10):
                    #     p.append(np.sqrt(dx[i]**2 + dy[i]**2+ dz[i]**2))
                    #     print('ade point%i:  ' + str(p[i]), i)
                    # ax.scatter(obs_trajpx[0:10], obs_trajpy[0:10], obs_trajpz[0:10], s=10, label="observed trajectory", c='red')
                    # ax.scatter(obs_trajpx[10:20], obs_trajpy[10:20], obs_trajpz[10:20], s=10, label="generated trajectory", c='blue')
                    # #ax.scatter(pred_traj_gtp.T[0], pred_traj_gtp.T[1], pred_traj_gtp.T[2], s=10, label="real trajectory",          c='orange')
                    # #ax.scatter(pred_traj_fake.T[0], pred_traj_fake.T[1], pred_traj_fake.T[2], s=10,    label="predict trajectory", c='blue')
                    # # print(pred_traj_fake)
                    # ax.legend()
                    # plt.savefig('books_read%i.png' % 1)

                    # print(decode(y[0].tolist()))
                    # print('---------------')

