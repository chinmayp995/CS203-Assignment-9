from cloudpickle import load
import numpy as np
import mlrun

class RFModel(mlrun.serving.V2ModelServer):
    def load(self):
        model_file, _ = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body: dict):
        feats = np.asarray(body['inputs'])
        results = self.model.predict(feats)
        return results.tolist()
