

def get_proto_targets(self, target, text_classes):

    text_target = [self.text_classes[i] for i in target]
    vec_target = [self.word2vec_model.wv[t] for t in text_target]
    return vec_target
