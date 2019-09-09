import torch
import torch.nn as nn
import torchvision


class Resnet(nn.Module):
    def __init__(self, projection_size=None, fine_tune=False):
        '''
        Construct a resnet101 model and exclude the last pooling and linear layer
        :param projection_size: Add an extra pooling layer to force the outputs
        have size (channels, projection_size, project_size). If None, do not add
        this layer
        :param fine_tune: Determine whether the model need to be fine-tune
        '''
        super(Resnet, self).__init__()
        resnet_base = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet_base.children())[: -2])

        self.projection_size = projection_size
        if projection_size is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(projection_size)

        if not fine_tune:
            # if we disable fine-tune, we need to freeze the parameters in resnet model
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, imgs):
        '''
        Propagate steps
        :param imgs: input images with expected shape (B, C, H, W), where B is batch
        size, C indicates channels, H and W are height and width respectively.
        :return: extracted features with size (B, 2048, H_new, W_new)
        '''
        features = self.resnet(imgs)
        if self.projection_size is not None:
            features = self.adaptive_pool(features)
        return features


class Attenrion(nn.Module):
    def __init__(self, feature_size, hidden_size, attention_size):
        super(Attenrion, self).__init__()
        self.feature_projection = nn.Linear(feature_size, attention_size)
        self.hidden_projection = nn.Linear(hidden_size, attention_size)
        self.attention_logit = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_states):
        '''
        Steps to calculate attention
        :param features: feature extracted from images with shape (batch_size, nums_of_features, feature_size)
        :param hidden_states: hidden states in LSTM with shape (batch_size, hidden_size)
        :return:
        '''
        # project the input features and hidden_states
        features_projected = self.feature_projection(features)  # shape: (batch_size, nums_of_features, attention_size)
        hidden_states_projected = self.hidden_projection(hidden_states)  # shape: (batch_size, attention_size)

        # shape: (batch_size, nums_of_features. attention_size)
        projection = features_projected + hidden_states_projected.unsqueeze(dim=1)
        projection = nn.ReLU()(projection)

        # shape: (batch_size, nums_of_features)
        logits = self.attention_logit(projection)
        logits = logits.squeeze(dim=2)

        attention_weight = self.softmax(logits)

        # shape: (batch_size, feature_size)
        output = features * attention_weight.unsqueeze(dim=2)
        output = output.sum(dim=1)

        return output, attention_weight


class DecoderAttCell(nn.Module):
    def __init__(self, feature_size, attention_size, embedd_size, hidden_size, vocab_size=None, pretrained_embedding=None):
        '''

        :param feature_size: size of extracted features
        :param attention_size: intermediate projection size used to calculated attention
        :param embedd_size: size of word embedding
        :param hidden_size: hidden size of LSTM
        :param pretrained_embedding: pre-trained word embedding matrix
        '''
        super(DecoderAttCell, self).__init__()
        if pretrained_embedding is not None:
            embedd_size = pretrained_embedding.shape[-1]
            self.embedding = self._load_embedding(pretrained_embedding=pretrained_embedding)
        else:
            assert vocab_size is not None and embedd_size is not None, \
                'vocab size and embedding size cannot be None if pre-trained embedding is not provided'
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedd_size)

        self.feature_size = feature_size
        self.attention = Attenrion(feature_size=feature_size, hidden_size=hidden_size, attention_size=attention_size)
        self.lstm = nn.LSTMCell(input_size=feature_size + embedd_size, hidden_size=hidden_size)
        self.init_h = nn.Linear(in_features=feature_size, out_features=hidden_size)
        self.init_c = nn.Linear(in_features=feature_size, out_features=hidden_size)
        self.f_beta = nn.Linear(in_features=hidden_size, out_features=feature_size)
        self.sigmoid = nn.Sigmoid()
        self.get_logits = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.get_logits.bias.data.fill_(0)
        self.get_logits.weight.data.uniform_(-0.1, 0.1)

    def forward(self, tokens, img_feature, h=None, c=None):
        word_features = self.embedding(tokens)

        feature_size = img_feature.shape[1]
        batch_size = img_feature.shape[0]
        assert feature_size == self.feature_size

        # reshape img_feature to have shape:
        # (batch_size, num_of_features/num_of_pixels, feature_size)
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(batch_size, -1, feature_size)

        if h is None and c is None:
            # if h and c is None, initialize them by using linear layer
            h, c = self.init_hidden(img_feature)

        # calculate attention base on image feature and prev hidden states
        attention_features, _ = self.attention(img_feature, h)
        gate = self.sigmoid(self.f_beta(h))
        attention_features = gate * attention_features

        lstm_input = torch.cat((word_features, attention_features), dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        logits = self.get_logits(h)
        log_probs = self.log_softmax(logits)
        return log_probs, logits, (h, c)



    def init_hidden(self, img_feature):
        img_feature = img_feature.mean(dim=1)
        h = self.init_h(img_feature)
        c = self.init_c(img_feature)
        return h, c

    def _load_embedding(self, pretrained_embedding):
        embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        return embedding
