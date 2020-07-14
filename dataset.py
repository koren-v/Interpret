import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, data_list, max_length, tokenizer):
        """
        :param data_list: list of 2 lists in order [title, text,  label]
        :param max_length: max model's input length
        :param tokenizer: model's tokenizer
        """

        self.data_list = data_list
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        tokenized_title = self.tokenizer.tokenize(self.data_list[0][index])
        tokenized_text = self.tokenizer.tokenize(self.data_list[1][index])

        tokenized_text = ['[CLS]'] + tokenized_title + ['[SEP]'] + tokenized_text + ['[SEP]']

        difference = len(tokenized_text) - self.max_length

        if difference > 0:  # too long
            attention_mask = [1]*self.max_length
            tokenized_text = tokenized_text[:self.max_length]
        else:
            attention_mask = [1]*len(tokenized_text) + [0]*-difference
            tokenized_text += ['[PAD]']*-difference

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        assert len(input_ids) == self.max_length

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        # if labels is available
        if len(self.data_list) == 3:
            labels = self.data_list[-1][index]
            labels = torch.tensor(labels)
            output = inputs, labels
        else:
            output = inputs

        return output

    def __len__(self):
        return len(self.data_list[0])
