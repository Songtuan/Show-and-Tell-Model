import torch
import torch.nn as nn
import allennlp.nn.beam_search as allen_beam_search

from modules.updown_cell import UpDownCell
from modules.FasterRCNN import FasterRCNN_Encoder
from modules.BeamSearch import BeamSearch
from functools import partial


class UpDownCaptioner(nn.Module):
    def __init__(self, vocab, encoder_out_dim=None, fine_tune=False, seq_length=15,
                 embed_dim=50, hidden_dim=512, attention_projection_size=512,
                 pre_trained_embedding=None, state_machine=None, beam_size=3):

        super(UpDownCaptioner, self).__init__()
        image_feature_size = encoder_out_dim if encoder_out_dim is not None else 1024
        vocab_size = len(vocab)
        self.vocab = vocab
        self.fine_tune = fine_tune
        self.state_machine = state_machine
        self.beam_size = beam_size
        self.img_feature_size = image_feature_size

        self.encoder = FasterRCNN_Encoder(fine_tune=fine_tune)
        self.decoder_cell = UpDownCell(image_feature_size=image_feature_size, embedding_size=embed_dim,
                                       hidden_size=hidden_dim, attention_projection_size=attention_projection_size)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'], reduction='sum')
        self.seq_length = seq_length

        if pre_trained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding).float()
            assert pre_trained_embedding.shape[1] == embed_dim
            assert pre_trained_embedding.shape[0] == len(vocab)
        else:
            vocab_size = len(vocab)
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    def _step(self, tokens, states, img_features):
        token_embedding = self.embedding(tokens)
        logits, states = self.decoder_cell(img_features, token_embedding, states)
        logits = self.fc(logits)
        log_probs = self.log_softmax(logits)
        return log_probs, logits, states

    def _decode_step(self, tokens, states, img_features):
        '''
        used in allennlp beam search
        Args:
            tokens(torch.Tensor): input token, shape: (batch_size, )
            states(dict[str, torch.Tensor]): input hidden state, shape: (batch_size, hidden_state)
            img_features(torch.Tensor): extracted image features through faster-rcnn,
                                        shape: (batch_size, num_boxes. feature_size)

        Returns:
            torch.Tensor: log probability distribution of each word in vocabulary
            dict[str, torch.Tensor]: hidden state produced by decoder cell
        '''
        if img_features.shape[0] != tokens.shape[0]:
            img_features = img_features.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
            batch_size, beam_size, num_boxes, image_feature_size = img_features.shape
            img_features = img_features.view(batch_size * beam_size, num_boxes, image_feature_size)
        token_embedding = self.embedding(tokens)
        logits, states = self.decoder_cell(img_features, token_embedding, states)
        logits = self.fc(logits)
        log_probs = self.log_softmax(logits)
        return log_probs, states

    def forward(self, imgs, captions=None):
        '''
        forward propagation of caption model
        Args:
            imgs(torch.Tensor): input image data, shape: (batch_size, channels, height, width)
            captions(torch.Tensor, optional): ground-truth captions, required only in training

        Returns:
            dict[str, torch.Tensor]: output of caption model
        '''
        batch_size = len(imgs)
        if not self.fine_tune:
            # if faster-rcnn do not require fine-tune
            # set it to eval mode so that we do not
            # have to pass ground-truth bounding box to it
            self.encoder.eval()

        # convert the batch data into list, which is required input format of faster-rcnn
        imgs = [imgs[i, :, :, :] for i in range(batch_size)]
        # shape (batch_size*100, encoder_out_dim), where 100 is the number of
        # proposals produced by RPN
        img_features, _ = self.encoder(imgs)
        # reshape to size (batch_size, 100, encoder_out_dim)
        img_features = img_features.view(batch_size, -1, self.img_feature_size)
        # img_features.double()
        assert self.img_feature_size == img_features.shape[-1]

        output_dict = {}

        if self.training:
            assert captions is not None, 'ground truth captions cannot be None when training'

            states = None
            max_length = captions.shape[-1]
            loss = 0

            for t in range(max_length):
                tokens = captions[:, t]
                log_probs, logits, states = self._step(tokens=tokens, states=states, img_features=img_features)
                loss += self.criterion(logits, captions[:, t])

            output_dict['loss'] = loss
            return output_dict
        else:
            if self.state_machine is None:
                # if state_machine is None, use normal beam search
                beam_search = allen_beam_search.BeamSearch(end_index=self.vocab['<end>'],
                                                           max_steps=self.seq_length, beam_size=self.beam_size,
                                                           per_node_beam_size=self.beam_size)
                init_tokens = torch.tensor([self.vocab['<start>']]).expand(batch_size).cuda()
                states = None
                step = partial(self._decode_step, img_features=img_features)
                top_k_preds, log_probs = beam_search.search(start_predictions=init_tokens, start_state=states,
                                                            step=step)
                preds = top_k_preds[:, 0, :]
                output_dict['seq'] = preds
                return output_dict

            done_beam = [[] for _ in range(batch_size)]
            beam_search = BeamSearch(beam_size=3, state_machine=self.state_machine,
                                     end_token_idx=self.vocab['<end>'],
                                     seq_length=self.seq_length, vocab=self.vocab)
            beam_num = len(self.state_machine.get_states())
            seq = torch.LongTensor(self.seq_length, batch_size).zero_()
            seq_logprobs = torch.FloatTensor(self.seq_length, batch_size)

            for k in range(batch_size):
                img_features_temp = img_features[k:k + 1, :, :].expand(beam_num * self.beam_size,
                                                                       img_features.shape[1],
                                                                       img_features.shape[2])
                init_tokens = torch.tensor([self.vocab['<start>']]).expand(self.beam_size * beam_num).cuda()
                states = None
                log_probs, _, states = self._step(tokens=init_tokens, states=states, img_features=img_features_temp)

                get_logprobs = partial(self.step, img_features=img_features_temp)

                done_beam[k] = beam_search.search(hidden_states=states, log_probs=log_probs, get_logprobs=get_logprobs)

                seq[:, k] = done_beam[k][0]['seq']
                seq_logprobs[:, k] = done_beam[k][0]['logps']

                output_dict['seq'] = seq
                output_dict['seq_logprobs'] = seq_logprobs
                return output_dict
