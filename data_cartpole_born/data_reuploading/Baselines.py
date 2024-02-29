import numpy as np

class LinearFeatureBaseline():
    def __init__(self, reg_coeff=1e-5):
        self.coeffs = None
        self.reg_coeff = reg_coeff

    def set_param_values(self, val):
        self.coeffs = val

    def features(self, states, rewards):
        o = np.clip(states, -10, 10)
        l = len(rewards)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, batch_states, batch_returns):
        featmat = np.concatenate([self.features(states,rewards) for states,rewards in zip(batch_states,batch_returns)])
        batch_returns = np.concatenate(batch_returns)

        reg_coeff = self.reg_coeff
        for _ in range(5):
            self.coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(batch_returns),rcond=None
            )[0]
            if not np.any(np.isnan(self.coeffs)):
                break
            reg_coeff *= 10

    def predict(self, states, rewards):
        if self.coeffs is None:
            return np.zeros(len(states))
        return self.features(states,rewards).dot(self.coeffs)#.astype("float32")
    
