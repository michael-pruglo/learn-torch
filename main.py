import torch


def main():
    t = torch.rand(6, 4)
    print(t.dtype)
    print(t, t.shape)
    print(t.device)
    print(t.layout)
    t = t.reshape(8, 3)
    print(t, t.shape)

    print("torch version", torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)


if __name__ == "__main__":
    main()
