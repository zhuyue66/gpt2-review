import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import tqdm
from torch.autograd import Variable


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--device', type=str, default='GPU',
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

    # 使用 GPT2Tokenizer 对输入进行编码
    text = "Yesterday, a man named Jack said he saw an alien,"
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    print('使用 GPT2Tokenizer 对输入进行编码')

    total_predicted_text = text
    n = 1000  # 预测过程的循环次数
    for _ in tqdm.trange(n):
        with torch.no_grad():
            tokens_tensor = Variable(tokens_tensor).to(device)
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = select_top_k(predictions, k=10)
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        total_predicted_text += tokenizer.decode(predicted_index)

        if '<|endoftext|>' in total_predicted_text:
            # 如果出现文本结束标志，就结束文本生成
            break

        indexed_tokens += [predicted_index]
        tokens_tensor = torch.tensor([indexed_tokens])

    print(total_predicted_text)


if __name__ == '__main__':
    main()
