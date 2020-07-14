import torch

from tqdm import tqdm

from saliency_interpreter import SaliencyInterpreter


class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)
    Registered as a `SaliencyInterpreter` with name "smooth-gradient".
    """
    def __init__(self,
                 model,
                 criterion,
                 tokenizer,
                 show_progress=True,
                 stdev=0.01,
                 num_samples=10):
        super().__init__(model, criterion, tokenizer, show_progress)
        # Hyperparameters
        self.stdev = stdev
        self.num_samples = num_samples
        self.show_progress = show_progress

    def saliency_interpret(self, test_dataloader):

        # Convert inputs to labeled instances
        predictions = self._get_prediction(test_dataloader)

        instances_with_grads = []

        iterator = tqdm(zip(*predictions), total=len(predictions[0])) \
            if self.show_progress else zip(*predictions)

        for probs, inputs, tokens in iterator:
            # Run smoothgrad
            labels = torch.argmax(probs, dim=1)
            grads = self._smooth_grads(labels, inputs)

            # sum by embeddings (scalar for each token)
            embedding_grads = grads.sum(dim=2)
            # norm for each sequence
            norms = torch.norm(embedding_grads, dim=1, p=1)
            # normalizing
            for i, norm in enumerate(norms):
                embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

            for i, embedding_grad in enumerate(embedding_grads):
                example_dict = dict()
                # as we do it by batches we has a padding so we need to remove it
                example_tokens = [t for t in tokens[i] if t != '[PAD]']
                example_dict['tokens'] = example_tokens
                example_dict['grad'] = embedding_grad.cpu().tolist()[:len(example_tokens)]
                example_dict['label'] = labels[i].item()
                example_dict['prob'] = probs[i].max().item()
                instances_with_grads.append(example_dict)

        return instances_with_grads

    def _register_forward_hook(self, stdev: float):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """

        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape).to(output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        embedding_layer = self.model.bert.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(self, label, inp):
        total_gradients = None
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            grads = self._get_gradients(label, inp)
            handle.remove()

            # Sum gradients
            if total_gradients is None:
                total_gradients = grads
            else:
                total_gradients = total_gradients + grads

        total_gradients /= self.num_samples

        return total_gradients
