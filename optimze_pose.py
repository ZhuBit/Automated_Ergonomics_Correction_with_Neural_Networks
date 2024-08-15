import os
import numpy as np
from utils import Setup_environment
Setup_environment()

from src.ergonomics_torch import Ergonomics_Torch
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from visualiser import visualize_poses_in_video,visualize_optimized_pose_for_defence
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
from src.ergonomics import RULA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

data_3d_path = 'data/intermedia/data_3d.npz'



if __name__ == '__main__':
    npz_files = glob.glob('data/npz/*.npz')

    # Coefficients for loss
    alpha = torch.tensor(0.4, dtype=torch.float32, requires_grad=True, device=device)  # Weight for ergo loss
    beta = torch.tensor(0.3, dtype=torch.float32, requires_grad=True, device=device)  # Weight for structural loss
    gamma = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)  # Weight for Discriminator loss

    for file in npz_files:
        print(f"Processing {file}...")
        loaded_data = np.load(file)
        initial_pose_3d = loaded_data['kps']


        file_name = os.path.basename(file)
        base_name = os.path.splitext(file_name)[0]  # Extract base name without extension
        print("Base Name:", base_name)

        print('Initial pose shape ', initial_pose_3d.shape)
        if isinstance(initial_pose_3d, np.ndarray):
            initial_pose_3d = torch.from_numpy(initial_pose_3d).float()
        initial_pose_3d = initial_pose_3d.to(device)
        initial_pose_3d.requires_grad_(True)

        # HumanPose GAN
        discriminator = HumanPoseDiscriminator().to(device)
        discriminator.load_state_dict(torch.load('models/discriminator.pth', map_location=device))
        discriminator.eval()

        optim_module = Ergonomics_Torch(device)
        optim_module.to(device)

        optimized_pose = optimize_pose(initial_pose_3d, optim_module, discriminator, base_name, alpha, beta, gamma)

        initial_pose_np = initial_pose_3d.cpu().detach().numpy()
        optimized_pose_np = optimized_pose.cpu().detach().numpy()

        # Call the evaluation function
        diff_list = evaluate_and_plot(initial_pose_np, optimized_pose_np, base_name)
        visualize_poses_in_video(initial_pose_np, optimized_pose, alpha.item(), beta.item(), gamma.item(), file)

        file_path = os.path.join('data/latest/optimised_npz', base_name)
        #np.savez(file_path, kps=optimized_pose_np)




def print_grad(named_tensor):
    tensor_name, tensor = named_tensor
    if tensor.grad is not None:
        print(f"Gradient for {tensor_name}: {tensor.grad}")
    else:
        print(f"No gradient for {tensor_name}")

def vector_difference(vector1, vector2):
    vector2 = vector2.cpu().numpy()
    vector1 = np.array(vector1)

    difference = vector1 - vector2

    return difference

def matrix_difference(matrix1, matrix2):
    if isinstance(matrix2, torch.Tensor):
        matrix2 = matrix2.cpu().numpy()
    elif not isinstance(matrix2, np.ndarray):
        matrix2 = np.array(matrix2)
    difference = matrix1 - matrix2

    # calculate the Euclidean norm (L2 norm) of the difference
    norm = np.linalg.norm(difference)

    return norm

def TMSE(optimized_poses):
    diffs = optimized_poses[1:] - optimized_poses[:-1]
    loss = torch.mean(diffs.pow(2))
    return loss
def plot_scores_comparison(scores_initial, scores_optim, shoulders_initial, shoulders_optim, trunk_initial, trunk_optim,
                           knee_initial, knee_optim, fps, file_name, differences_list):
    # Directory Creation
    output_dir = 'data/defence'
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Figure Initialization
    plt.figure(figsize=(10, 16))

    # Determine the shortest length for each comparison to avoid dimension mismatch
    scores_length = min(len(scores_initial), len(scores_optim))
    shoulders_length = min(len(shoulders_initial), len(shoulders_optim))
    trunk_length = min(len(trunk_initial), len(trunk_optim))
    knee_length = min(len(knee_initial), len(knee_optim))

    # Timestamps Calculation
    max_length = max(scores_length, shoulders_length, trunk_length, knee_length)
    if fps:
        timestamps = np.linspace(0, max_length / fps, num=max_length)
    else:
        timestamps = np.arange(max_length)

def evaluate_and_plot(initial_pose_3d, optimized_pose, file_name):
    # Assume RULA and plotting functions are already defined or imported
    rula_eval_initial = RULA(initial_pose_3d)
    scores_initial, initial_shoulders, initial_trunk, initial_knee = rula_eval_initial.compute_scores()

    rula_eval_optimized = RULA(optimized_pose)
    scores_optimized, optimized_shoulders, optimized_trunk, optimized_knee = rula_eval_optimized.compute_scores()

    differences_list = [item2 - item1 for item1, item2 in zip(scores_initial, scores_optimized)]

    plot_scores_comparison(scores_initial, scores_optimized,
                           shoulders_initial=initial_shoulders, shoulders_optim=optimized_shoulders,
                           trunk_initial=initial_trunk, trunk_optim=optimized_trunk,
                           knee_initial=initial_knee, knee_optim=optimized_knee,
                           file_name=file_name, differences_list=differences_list, fps=30)

    return  differences_list


def optimize_pose(pose_3d_initial, ergonomic_torch, discriminator, file, alpha, beta, gamma):
    lr = 0.01
    num_steps = 5
    print_interval = 20

    pose_3d = pose_3d_initial.clone().detach().requires_grad_(True)
    print('Intial Pose: ', pose_3d_initial.shape, ' copy: ', pose_3d.shape)

    optimizer = optim.Adam([pose_3d], lr=lr)
    #optimizer = optim.SGD([pose_3d], lr=lr)
    #optimizer = optim.RMSprop([pose_3d], lr=lr)

    for step in range(num_steps):
        print(step, '--------------------------------')
        optimizer.zero_grad()
        ergo_loss = ergonomic_torch(pose_3d)
        structural_loss = TMSE(pose_3d)
        print('ergo_loss: ', ergo_loss)
        print('structural_loss: ', structural_loss)

        # Discriminator loss
        discriminator_losses = []
        for frame in pose_3d:
            frame_loss = -torch.log(discriminator(frame.flatten()))
            discriminator_losses.append(frame_loss)

        # Aggregate discriminator losses
        discriminator_loss = torch.sum(torch.stack(discriminator_losses))

        print('discriminator_loss: ', discriminator_loss.item())

        # Compute the composite loss
        composite_loss = alpha * ergo_loss + beta * structural_loss + gamma * discriminator_loss

        loss = torch.sum(composite_loss)

        loss.backward()
        optimizer.step()

        if step % print_interval == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    return pose_3d.detach()

class HumanPoseDiscriminator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HumanPoseDiscriminator, self).__init__()
        # (17 keypoints * 3 coordinates)
        self.model = nn.Sequential(
            nn.Linear(51, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, frame):
        validity = self.model(frame)
        return validity