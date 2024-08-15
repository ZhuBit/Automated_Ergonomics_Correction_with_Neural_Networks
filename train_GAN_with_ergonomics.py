import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from HumanPoseGAN.dataset import HPFrameDataset
from HumanPoseGAN.visualizer_html import visualize_and_save_frame_with_belief
from src.ergonomics_torch import *

class HumanPoseDiscriminator(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HumanPoseDiscriminator, self).__init__()
        # (17 keypoints * 3 coordinates)
        self.model = nn.Sequential(
            nn.Linear(65, 128),
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
        self.device = torch.device("cuda")
        # (17 keypoints * 3 coordinates)
        """  self.model = nn.Sequential(
            nn.Linear(51, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
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
        )"""

    def forward(self, batch_frames):
        print('batch_frames', batch_frames.shape)
        reshaped_frames = batch_frames.reshape(-1, 17, 3)
        print("reshaped_frames", reshaped_frames.shape)
        angles = self.accumulate_frames_angles(reshaped_frames)
        print('angles', angles.shape)
        flat_frames = reshaped_frames.reshape(batch_frames.shape[0], -1)
        enriched_frames = torch.cat((flat_frames, angles), dim=1)
        print('enriched_frames', enriched_frames.shape)

        validity = self.model(enriched_frames)
        return validity
    def accumulate_frames_angles(self, pose_3d):

        """
        computes the angles between body parts as specified by the RULA worksheet
        :return: angles between body parts
        """
        all_angles = []

        for ind in range(len(pose_3d)):
            angles_frame = []

            pose = self.transform_pose(pose_3d[ind,...].clone()) # deep copy

            # left shoulder angles
            shoulder_left_z = self.calculate_z(pose, 11, 12) - 10
            angles_frame.append(shoulder_left_z)
            shoulder_left_shoulderline = self.calculate_angle(pose, 12, 11, 14)
            angles_frame.append(shoulder_left_shoulderline)

            # right shoulder angles
            angle_shoulder_right_z = self.calculate_z(pose, 14, 15) - 10
            angles_frame.append(angle_shoulder_right_z)
            angle_shoulder_right_shoulderline = self.calculate_angle(pose, 15, 14, 11)
            angles_frame.append(angle_shoulder_right_shoulderline)

            # elbow angles
            angle_elbow_left = 180 - self.calculate_angle(pose, 13, 12, 11)
            angles_frame.append(angle_elbow_left)
            angle_elbow_right = 180 - self.calculate_angle(pose, 14, 15, 16)
            angles_frame.append(angle_elbow_right)

            # knee angles
            angle_knee_left = 180 - self.calculate_angle(pose, 4, 5, 6)
            angles_frame.append(angle_knee_left)
            angle_knee_right = 180 - self.calculate_angle(pose, 1, 2, 3)
            angles_frame.append(angle_knee_right)

            # trunk
            angle_stoop = 180 - self.calculate_z(pose, 0, 8)
            angles_frame.append(angle_stoop)
            angle_trunk_twist = self.calculate_twist(pose, 1, 4, 11, 14)
            angles_frame.append(angle_trunk_twist)
            angle_trunk_sidebending = self.calculate_twist(pose, 1, 4, 0, 7)
            angles_frame.append(angle_trunk_sidebending)

            # neck
            angle_neck = 180 - self.calculate_z(pose, 8, 10)
            angles_frame.append(angle_neck)
            angle_neck_sidebending = self.calculate_twist(pose, 11, 14, 8, 10)
            angles_frame.append(angle_neck_sidebending)
            angle_neck_twist = self.calculate_twist(pose, 11, 14, 9, 10)
            angles_frame.append(angle_neck_twist)

            #angles_frame_tensor = torch.tensor(angles_frame, requires_grad=True)
            angles_frame_tensor = torch.stack(angles_frame)
            all_angles.append(angles_frame_tensor)

        return torch.stack(all_angles, dim=0)

    def transform_pose(self, pose):
        """
        takes the original pose and transposes it from XZY to XYZ and flips the Z axes
        :param pose: 3D keypoints
        :return:
        """
        # Use torch's transpose method instead of np.transpose
        new_pose = torch.transpose(pose, 0, 1)

        # Swap the second and third dimensions
        temp = new_pose[1, :].clone()
        new_pose[1, :] = new_pose[2, :].clone()
        new_pose[2, :] = temp

        # Flip the Z axis
        new_pose[2, :] = new_pose[2, :].neg()

        return new_pose

    def calculate_z(self, pose, joint1, joint2):
        a = pose[:, joint1]
        b = pose[:, joint2]

        ba = a - b
        bc = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        """cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(cosine_angle)"""
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        angle_degrees = angle * (180 / torch.pi)

        return angle_degrees

    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]

        ba = a - b
        bc = c - b

        # Ensure ba and bc are 1-dimensional or use torch.matmul for multi-dimensional tensors
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]
        d = pose[:, joint4]

        ba = a - b
        dc = d - c

        # Use torch.sum for dot product to handle tensors of any shape
        cosine_angle = torch.sum(ba * dc) / (torch.norm(ba) * torch.norm(dc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)
class HumanPoseGenerator(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(HumanPoseGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),

            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 51),  # 17 keypoints * 3 coordinates
            nn.Tanh()
        )
        """super(HumanPoseGenerator, self).__init__()
         6x self.model = nn.Sequential(
            nn.Linear(100, 256),

            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 51),
            nn.Tanh()
        )"""

        """self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 51),  # 17 keypoints * 3 coordinates
            # nn.Tanh()
        )"""

    def forward(self, z):
        pose = self.model(z)
        return pose
    def accumulate_frames_angles(self, pose_3d):

        """
        computes the angles between body parts as specified by the RULA worksheet
        :return: angles between body parts
        """
        all_angles = []

        for ind in range(len(pose_3d)):
            angles_frame = []

            pose = self.transform_pose(pose_3d[ind,...].clone()) # deep copy

            # left shoulder angles
            shoulder_left_z = self.calculate_z(pose, 11, 12) - 10
            angles_frame.append(shoulder_left_z)
            shoulder_left_shoulderline = self.calculate_angle(pose, 12, 11, 14)
            angles_frame.append(shoulder_left_shoulderline)

            # right shoulder angles
            angle_shoulder_right_z = self.calculate_z(pose, 14, 15) - 10
            angles_frame.append(angle_shoulder_right_z)
            angle_shoulder_right_shoulderline = self.calculate_angle(pose, 15, 14, 11)
            angles_frame.append(angle_shoulder_right_shoulderline)

            # elbow angles
            angle_elbow_left = 180 - self.calculate_angle(pose, 13, 12, 11)
            angles_frame.append(angle_elbow_left)
            angle_elbow_right = 180 - self.calculate_angle(pose, 14, 15, 16)
            angles_frame.append(angle_elbow_right)

            # knee angles
            angle_knee_left = 180 - self.calculate_angle(pose, 4, 5, 6)
            angles_frame.append(angle_knee_left)
            angle_knee_right = 180 - self.calculate_angle(pose, 1, 2, 3)
            angles_frame.append(angle_knee_right)

            # trunk
            angle_stoop = 180 - self.calculate_z(pose, 0, 8)
            angles_frame.append(angle_stoop)
            angle_trunk_twist = self.calculate_twist(pose, 1, 4, 11, 14)
            angles_frame.append(angle_trunk_twist)
            angle_trunk_sidebending = self.calculate_twist(pose, 1, 4, 0, 7)
            angles_frame.append(angle_trunk_sidebending)

            # neck
            angle_neck = 180 - self.calculate_z(pose, 8, 10)
            angles_frame.append(angle_neck)
            angle_neck_sidebending = self.calculate_twist(pose, 11, 14, 8, 10)
            angles_frame.append(angle_neck_sidebending)
            angle_neck_twist = self.calculate_twist(pose, 11, 14, 9, 10)
            angles_frame.append(angle_neck_twist)

            #angles_frame_tensor = torch.tensor(angles_frame, requires_grad=True)
            angles_frame_tensor = torch.stack(angles_frame)
            all_angles.append(angles_frame_tensor)

        return torch.stack(all_angles, dim=0)

    def transform_pose(self, pose):
        """
        takes the original pose and transposes it from XZY to XYZ and flips the Z axes
        :param pose: 3D keypoints
        :return:
        """
        # Use torch's transpose method instead of np.transpose
        new_pose = torch.transpose(pose, 0, 1)

        # Swap the second and third dimensions
        temp = new_pose[1, :].clone()
        new_pose[1, :] = new_pose[2, :].clone()
        new_pose[2, :] = temp

        # Flip the Z axis
        new_pose[2, :] = new_pose[2, :].neg()

        return new_pose

    def calculate_z(self, pose, joint1, joint2):
        a = pose[:, joint1]
        b = pose[:, joint2]

        ba = a - b
        bc = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        """cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(cosine_angle)"""
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        angle_degrees = angle * (180 / torch.pi)

        return angle_degrees

    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]

        ba = a - b
        bc = c - b

        # Ensure ba and bc are 1-dimensional or use torch.matmul for multi-dimensional tensors
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]
        d = pose[:, joint4]

        ba = a - b
        dc = d - c

        # Use torch.sum for dot product to handle tensors of any shape
        cosine_angle = torch.sum(ba * dc) / (torch.norm(ba) * torch.norm(dc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data_path = 'data/npz'


    output_dir = ("HumanPoseGAN/data/outputs/angles+4GLayers+4DLay+0,0,3drDisc+64btch+Than+NoDecayG+NoDropGEner")
    #output_dir = "data/outputs/TEST"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 64
    num_epochs = 1000

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    noise_factor = 0.02
    apply_noise_prob = 0.5 # How often do noises

    train_dataset = HPFrameDataset(train_data_path)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset = HPFrameDataset(test_data_path)


    generator = HumanPoseGenerator().to(device)
    discriminator = HumanPoseDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999), weight_decay=1e-5)
    adversarial_loss = torch.nn.BCELoss()

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    generator_losses = []
    discriminator_losses = []
    avg_generator_losses = []
    avg_discriminator_losses = []

    real_accuracies = []
    fake_accuracies = []

    learning_rates_G = []
    learning_rates_D = []

    discriminator_scores_real = []
    discriminator_scores_generated = []

    l1_lambda = 1e-5

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        sum_generator_loss = 0
        sum_discriminator_loss = 0
        total_batches = 0
        epoch_discriminator_losses = discriminator_losses[epoch * total_batches: (epoch + 1) * total_batches]
        for i, (item, _) in enumerate(dataloader):
            print("Batch: {}".format(i))
            item = item.to(device)
            valid = torch.ones((item.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((item.size(0), 1), requires_grad=False).to(device)

            """if torch.rand(1).item() < apply_noise_prob:
                for n in range(item.shape[0]):  # Iterate over each item in the batch
                    num_keypoints_to_noise = torch.randint(0, min(5, item.shape[1] // 3),
                                                           (1,)).item() 
                    keypoints_indices = torch.randperm(item.shape[1] // 3)[
                                        :num_keypoints_to_noise]  # Adjusted for flattened structure
                    for kp in keypoints_indices:
                        start_index = kp * 3
                        end_index = start_index + 3
                        # Adjust noise addition for a 2D tensor
                        item[n, start_index:end_index] += noise_factor * torch.randn_like(
                            item[n, start_index:end_index]).to(device)"""

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(item.shape[0], 100).to(device)
            generated_item = generator(z)

            g_loss = adversarial_loss(discriminator(generated_item), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(item), valid)
            fake_loss = adversarial_loss(discriminator(generated_item.detach()), fake)

            l1_loss = 0
            for param in discriminator.parameters():
                l1_loss += torch.sum(torch.abs(param))
            d_loss = (real_loss + fake_loss) / 2 + l1_lambda * l1_loss

            d_loss.backward()
            optimizer_D.step()

            with torch.no_grad():
                real_predictions = discriminator(item)
                fake_predictions = discriminator(generated_item.detach())

                # Threshold predictions for binary classification
                real_predictions = real_predictions > 0.5
                fake_predictions = fake_predictions > 0.5

                real_accuracy = (real_predictions > 0.5).float().mean().item()
                fake_accuracy = (fake_predictions <= 0.5).float().mean().item()

                real_accuracies.append(real_accuracy)
                fake_accuracies.append(fake_accuracy)

                discriminator_scores_real.extend(real_predictions.cpu().numpy())
                discriminator_scores_generated.extend(fake_predictions.cpu().numpy())

                total_samples += item.size(0) * 2  # real and fake samples
                correct_predictions += (real_predictions == valid).sum().item() + (
                        fake_predictions == fake).sum().item()

                true_positives += real_predictions.sum().item()
                total_predicted_positives += real_predictions.sum().item() + fake_predictions.sum().item()

                if epoch % 10 == 0 and i == 0 and epoch > 10:
                    with torch.no_grad():
                        real_predictions = discriminator(item)
                        fake_predictions = discriminator(generated_item.detach())

                        # Calculate binary predictions and probabilities
                        real_pred_binary = (real_predictions > 0.5).float()
                        fake_pred_binary = (fake_predictions > 0.5).float()

                        # Select a single sample for visualization
                        sample_real = item[0]  # First item in the batch for real
                        sample_generated = generated_item[0]  # First item in the generated batch

                        real_pred_for_viz = real_pred_binary[0].item()  # Binary prediction for real sample
                        fake_pred_for_viz = fake_pred_binary[0].item()  # Binary prediction for fake sample

                        real_prob_for_viz = real_predictions[0].item()  # Probability for real sample
                        fake_prob_for_viz = fake_predictions[0].item()  # Probability for fake sample

                        # Visualization calls
                        visualize_and_save_frame_with_belief(sample_real.cpu().reshape(17, 3), f"Epoch {epoch} Real",
                                                             epoch, "real",
                                                             real_pred_for_viz, real_prob_for_viz, output_dir)
                        visualize_and_save_frame_with_belief(sample_generated.cpu().reshape(17, 3),
                                                             f"Epoch {epoch} Generated", epoch,
                                                             "generated", fake_pred_for_viz, fake_prob_for_viz,
                                                             output_dir)
            sum_generator_loss += g_loss.item()
            sum_discriminator_loss += d_loss.item()
            total_batches += 1

            # Add the individual losses to their respective lists (if you're tracking per batch losses)
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
            learning_rates_G.append(optimizer_G.param_groups[0]['lr'])
            learning_rates_D.append(optimizer_D.param_groups[0]['lr'])
            accuracy = correct_predictions / total_samples
            precision = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0
            print(f"Epoch {epoch}")
            print(f"Epoch {epoch} Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Accuracy: {accuracy:.2f}] [Precision: {precision:.2f}]")

        average_generator_loss = sum_generator_loss / total_batches
        average_discriminator_loss = sum_discriminator_loss / total_batches
        avg_generator_losses.append(average_generator_loss)
        avg_discriminator_losses.append(average_discriminator_loss)

    generator_model_path = os.path.join(output_dir, 'generator.pth')
    discriminator_model_path = os.path.join(output_dir, 'discriminator.pth')

    torch.save(generator.state_dict(), generator_model_path)
    torch.save(discriminator.state_dict(), discriminator_model_path)

    plt.figure(figsize=(15, 10))

    # Generator Loss vs. Discriminator Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Generator Loss vs. Discriminator Loss")

    # Real vs. Fake Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(real_accuracies, label="Real Accuracy")
    plt.plot(fake_accuracies, label="Fake Accuracy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Real vs. Fake Accuracy")

    # Learning Rates plot
    plt.subplot(2, 3, 3)
    plt.plot(learning_rates_G, label="Generator Learning Rate")
    plt.plot(learning_rates_D, label="Discriminator Learning Rate")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.title("Generator and Discriminator Learning Rates")

    # Discriminator Scores Distribution plot
    plt.subplot(2, 3, 4)
    discriminator_scores_real = np.array(discriminator_scores_real).flatten().astype(int)
    discriminator_scores_generated = np.array(discriminator_scores_generated).flatten().astype(int)
    plt.hist(discriminator_scores_real, bins=50, alpha=0.5, label="Real Scores", color='blue')
    plt.hist(discriminator_scores_generated, bins=50, alpha=0.5, label="Generated Scores", color='red')
    plt.legend()
    plt.xlabel("Discriminator Scores")
    plt.ylabel("Frequency")
    plt.title("Discriminator Scores Distribution")

    # Training Loss Curves vs Epochs
    epochs = list(range(1, num_epochs + 1))
    plt.subplot(2, 3, 5)
    plt.plot(epochs, avg_generator_losses, label="Generator Loss", marker='o')
    plt.plot(epochs, avg_discriminator_losses, label="Discriminator Loss", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    description_text = (f'Batch size: {batch_size}\n'
                        f'Num epochs: {num_epochs}\n'
                        f'Gen Loss: {avg_generator_losses[-1]:.2f}, Discrm Loss: {avg_discriminator_losses[-1]:.2f}\n'
                        f'Real Acc: {real_accuracies[-1] * 100:.2f}%, Fake Acc: {fake_accuracies[-1] * 100:.2f}%')

    plt.text(0.95, 0.05, description_text,
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()

    plt.tight_layout()
    comparison_plot_filename = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_plot_filename)

    plt.show()
    plt.close()






