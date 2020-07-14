import torch
from torch.nn.functional import softmax

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt


class SaliencyInterpreter:
    def __init__(self,
                 model,
                 criterion,
                 tokenizer,
                 show_progress=True):

        self.model = model
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_progress = show_progress

    def _get_prediction(self, test_dataloader):

        model_inputs = []
        input_tokens = []
        predictions = []
        model = self.model.to(self.device)
        model.eval()

        iterator = tqdm(test_dataloader) if self.show_progress else test_dataloader

        # for inputs in test_dataloader:
        for inputs in iterator:

            # collecting inputs, as they will be used in _get_gradients
            # and tokens to correspond them method output

            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get("attention_mask")

            tokens = [
                self.tokenizer.convert_ids_to_tokens(input_ids_)
                for input_ids_ in input_ids
            ]
            input_tokens.append(tokens)
            model_inputs.append(inputs)

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            predictions.append(softmax(outputs, dim=-1))

        return predictions, model_inputs, input_tokens

    def _get_gradients(self, label, inp):
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        input_ids = inp.get('input_ids').to(self.device)
        attention_mask = inp.get("attention_mask").to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        batch_losses = self.criterion(outputs, label)
        loss = torch.mean(batch_losses)

        self.model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return embedding_gradients[0]

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        Used to save the gradients of the embeddings for use in get_gradients()
        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = self.model.bert.embeddings
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    @staticmethod
    def colorize(instance, skip_special_tokens=False, special_tokens=['[CLS]', '[SEP]']):
        word_cmap = matplotlib.cm.Blues
        prob_cmap = matplotlib.cm.Greens
        template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
        colored_string = ''
        for word, color in zip(instance['tokens'], instance['grad']):
            if word in special_tokens and skip_special_tokens:
                continue
            # handle wordpieces
            word = word.replace("##", "") if "##" in word else ' ' + word
            color = matplotlib.colors.rgb2hex(word_cmap(color)[:3])
            colored_string += template.format(color, word)
        colored_string += template.format(
            0, "    Label: {} (".format(instance['label'])
        )
        prob = instance['prob']
        color = matplotlib.colors.rgb2hex(prob_cmap(prob)[:3])
        colored_string += template.format(
            color, "{:.2f}%".format(instance['prob']*100)
        ) + ')'
        return colored_string
