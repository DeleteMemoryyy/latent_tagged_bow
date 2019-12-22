from dataloader import Dataloader
import nltk


if __name__ == '__main__':
    print(nltk.find("."))
    loader = Dataloader()
    loader.load()
    loader.spilt_train_test()
    print(len(loader.train_tokens))