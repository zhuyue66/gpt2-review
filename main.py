import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import tqdm
from torch.autograd import Variable
import time
from torch.utils.data import DataLoader, TensorDataset


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index

def train():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--device', type=str, default='NPU',
                        help='device: CPU, GPU or NPU')
    args = parser.parse_args()

    if args.device == "CPU":
        device = torch.device("cpu")
    elif args.device == "GPU":
        device = torch.device("cuda")
    elif args.device == "NPU":
        device = torch.device("npu:0")
    else:
        raise Exception('Invalid device : {}'.format(args.device))
    print(device)

    # 载入预训练模型的分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('载入预训练模型的分词器')

    # 读取 GPT-2 预训练模型
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    print('读取 GPT-2 预训练模型')

    # 加载数据集
    with open('./data/romeo_and_juliet.txt', 'r') as f:
        dataset = f.read()

    # 编码
    indexed_text = tokenizer.encode(dataset)
    del (dataset)

    dataset_cut = []
    for i in range(len(indexed_text) // 512):
        # 将字符串分段成长度为 512
        dataset_cut.append(indexed_text[i * 512:i * 512 + 512])
    del (indexed_text)

    dataset_tensor = torch.tensor(dataset_cut)

    # 构建数据集和数据迭代器，设定 batch_size 大小为 2
    train_set = TensorDataset(dataset_tensor,
                              dataset_tensor)  # 标签与样本数据相同
    train_loader = DataLoader(dataset=train_set,
                              batch_size=8,
                              shuffle=False)

    pre = time.time()
    epoch = 3000  # 循环学习 3000 次
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

    for i in tqdm.trange(epoch):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # data to device
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad()
            result = model(data, labels=target)
            total_loss += result['loss']
            result['loss'].backward()
            optimizer.step()
            if batch_idx == len(train_loader) - 1:
                # 在每个 Epoch 的最后输出一下结果
                print('average loss:', total_loss / len(train_loader))
    print('训练时间：', time.time() - pre)

if __name__ == '__main__':
    train()