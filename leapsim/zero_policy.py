"""
A zero policy that outputs zero actions to hold the initial pose
"""
import torch

class ZeroPolicy:
    def __init__(self, num_actions=16):
        self.num_actions = num_actions
        
    def __call__(self, obs):
        """Return zero actions"""
        if isinstance(obs, torch.Tensor):
            return torch.zeros((obs.shape[0], self.num_actions), device=obs.device)
        else:
            return torch.zeros((1, self.num_actions))
            
    def eval(self):
        pass
        
    def to(self, device):
        return self

# Create a fake checkpoint that returns our zero policy
def load_checkpoint(path):
    return ZeroPolicy()

if __name__ == "__main__":
    # Save a dummy checkpoint
    policy = ZeroPolicy()
    torch.save(policy, "zero_policy.pth")