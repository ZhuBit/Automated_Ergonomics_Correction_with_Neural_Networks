import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.animation as animation
from datetime import datetime
# from common.visualization import read_video, get_fps, get_resolution

# def downsample_tensor(X, factor):
#     length = X.shape[0]//factor * factor
#     return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)
#
# def read_video_test(filename):
#     w, h = get_resolution(filename)
#
#     command = ['ffmpeg',
#                '-i', filename,
#                '-f', 'image2pipe',
#                '-pix_fmt', 'bgr24',
#                '-vsync', '0',
#                '-loglevel', 'quiet',
#                '-vcodec', 'rawvideo', '-']
#
#     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
#     i = 0
#     while True:
#         i += 1
#         data = pipe.stdout.read(w * h * 3)
#         if not data:
#             break
#         yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3)), str(i - 1).zfill(5)
#
#
# def read_video_custom(filename):
#     w, h = get_resolution(filename)
#
#     command = ['ffmpeg',
#                '-i', filename,
#                '-f', 'image2pipe',
#                '-pix_fmt', 'bgr24',
#                '-vsync', '0',
#                '-loglevel', 'quiet',
#                '-vcodec', 'rawvideo', '-']
#
#     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
#     while True:
#         data = pipe.stdout.read(w * h * 3)
#         if not data:
#             break
#         yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

#
# def render_animation_test(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
#                           limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
#     """
#     TODO
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     plt.ioff()
#     fig = plt.figure(figsize=(size * (1 + len(poses)), size))
#     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
#     ax_in.get_xaxis().set_visible(False)
#     ax_in.get_yaxis().set_visible(False)
#     ax_in.set_axis_off()
#     ax_in.set_title('Input')
#
#     ax_3d = []
#     lines_3d = []
#     trajectories = []
#     radius = 1.7
#     for index, (title, data) in enumerate(poses.items()):
#         ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         ax.set_xlim3d([-radius/2, radius/2])
#         ax.set_zlim3d([0, radius])
#         ax.set_ylim3d([-radius/2, radius/2])
#         try:
#             ax.set_aspect('equal')
#         except NotImplementedError:
#             ax.set_aspect('auto')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 7.5
#         ax.set_title(title)  # , pad=35
#         ax_3d.append(ax)
#         lines_3d.append([])
#         trajectories.append(data[:, 0, [0, 1]])
#     poses = list(poses.values())
#
#     # # Decode video
#     # if input_video_path is None:
#     #     # Black background
#     #     all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
#     # else:
#     #     # Load video using ffmpeg
#     #     all_frames = []
#     #     for frame_i, im in enumerate(read_video(input_video_path)):  # skip=input_video_skip, limit=limit
#     #         all_frames.append(im[0])
#     #     #effective_length = min(keypoints.shape[0], len(all_frames))
#     #     #all_frames = all_frames[:effective_length]
#     #
#     #     keypoints = keypoints[input_video_skip:]  # todo remove
#     #     for idx in range(len(poses)):
#     #         poses[idx] = poses[idx][input_video_skip:]
#     #
#     #     if fps is None:
#     #         fps = get_fps(input_video_path)
#
#     # if downsample > 1:
#     #     #keypoints = downsample_tensor(keypoints, downsample)
#     #     #all_frames = all_frames[::downsample]
#     #     all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
#     #     # for idx in range(len(poses)):
#     #     #     poses[idx] = downsample_tensor(poses[idx], downsample)
#     #     #     trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
#     #     fps /= downsample
#
#     initialized = False
#     image = None
#     lines = []
#     points = None
#
#     # print("length frames")
#     # print(len(all_frames))
#     # print("limit is:")
#     # print(all_frames[0].shape)
#
#     if limit < 1:
#         limit = len(all_frames)
#     else:
#         limit = min(limit, len(all_frames))
#
#     parents = skeleton.parents()  # [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12, 16]
#
#     def update_video(i):
#         nonlocal initialized, image, lines, points
#
#         for n, ax in enumerate(ax_3d):
#             ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#             ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
#         # print(n, ax)
#         # print(radius)
#         # print(trajectories[n][i, 0])
#         # print(i)
#
#         # print([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#
#         # ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#         # ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
#
#         # Update 2D poses
#         joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
#         colors_2d = np.full(17, 'black')  # keypoints.shape[1]
#         colors_2d[joints_right_2d] = 'red'
#
#         # print(colors_2d)
#         # print(keypoints.shape)
#
#         if not initialized:
#             print(all_frames[i].shape)
#             image = ax_in.imshow(all_frames[i], aspect='equal')
#
#             for j, j_parent in enumerate(parents):
#
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
#                     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
#
#                 col = 'red' if j in skeleton.joints_right() else 'black'
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
#                                                [pos[j, 1], pos[j_parent, 1]],
#                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
#
#             points = ax_in.scatter(*keypoints[i].T, 10, edgecolors='white', zorder=10)  # color=colors_2d,
#
#             initialized = True
#         else:
#             image.set_data(all_frames[i])
#
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
#
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]  # [n][i]
#
#                     # print(pos.shape)
#
#                     lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
#                     lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
#                     lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
#
#             points.set_offsets(keypoints[i])
#
#         print('{}/{}      '.format(i, limit), end='\r')
#
#     fig.tight_layout()
#
#     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
#     if output.endswith('.mp4'):
#         Writer = writers['ffmpeg']
#         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#         anim.save(output, writer=writer)
#     elif output.endswith('.gif'):
#         anim.save(output, dpi=80, writer='imagemagick')
#     else:
#         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#     plt.close()


def render_animation_custom(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, size=6, input_video=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))

    if input_video is None:
        render_2D = False
    else:
        render_2D = True

        ax_in = fig.add_subplot(1, 1 + len(poses), 1)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()
        ax_in.set_title('2D Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        if render_2D == True:
            ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        else:
            ax = fig.add_subplot(1, 1 + len(poses), (index + 1, index + 2), projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video is None:
        fps = keypoints_metadata['video_metadata']['data_2d']['fps']

        # Black background
        #all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
        #render_2D = False
    else:
        fps = input_video.fps

    # if downsample > 1:
    #     #keypoints = downsample_tensor(keypoints, downsample)
    #     if render_2D == True:
    #         all_frames = all_frames[::downsample]
    #     #all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8') #memory issues as the whole video has to be loaded to RAM...
    #     #for idx in range(len(poses)):
    #     #    poses[idx] = downsample_tensor(poses[idx], downsample)
    #     #    trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
    #     fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = poses[0].shape[0]#len(all_frames)
    else:
        limit = min(limit, poses[0].shape[0])#len(all_frames))

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            #render 2D video
            if render_2D:
                image = ax_in.imshow(input_video.frames[i][...,[2,1,0]], aspect='equal')
                points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            ###### ///////////////////////////////////////////////////////////
            # ax.scatter(poses[0][0, 1, 0], poses[0][0, 1, 1], 0, c='r', marker='o') #(255,0,0) poses[0][0, 1, 2]
            # ax.scatter(poses[0][0, 4, 0], poses[0][0, 4, 1], 0, c='r', marker='o') #poses[0][0, 4, 2]

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))


            initialized = True
        else:
            # render 2D video
            if render_2D:
                image.set_data(input_video.frames[i][...,[2,1,0]])
                points.set_offsets(keypoints[i])

            ###### ///////////////////////////////////////////////////////////
            #ax.scatter(poses[0][i, 1, 0], poses[0][i, 1, 1], 0, c='r', marker='o')  # (255,0,0) poses[0][0, 1, 2]
            #ax.scatter(poses[0][i, 4, 0], poses[0][i, 4, 1], 0, c='r', marker='o')  # poses[0][0, 4, 2]

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')



        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):# or output.endswith('.avi'):
        Writer = writers['ffmpeg'] #writers['pillow'] #
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
    print('Video saved')

def visualize_poses_in_video(original_pose, optimized_pose, alpha, beta, gamma, file_name):
    num_frames = original_pose.shape[0]

    # Define pairs of keypoints to connect
    pairs = [(0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10), (4, 5), (1, 2), (5, 6), (2, 3),
             (8, 11), (8, 14), (11, 12), (14, 15), (12, 13), (15, 16)]

    # Initialize plot with two subplots
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.view_init(elev=7, azim=50)
    ax2.view_init(elev=7, azim=50)

    # Function to update each frame in the animation
    def update_graph(num, original_pose, optimized_pose, pairs):
        ax1.clear()
        ax2.clear()
        original_pose_np = original_pose
        optimized_pose_np = optimized_pose.detach().cpu().numpy()
        # Plot for original pose in red
        x, y, z = original_pose_np[num, :, 0], original_pose_np[num, :, 1], original_pose_np[num, :, 2]
        ax1.scatter(x, y, z, color='red')
        for pair in pairs:
            ax1.plot([x[pair[0]], x[pair[1]]], [y[pair[0]], y[pair[1]]], [z[pair[0]], z[pair[1]]], color='red')
        ax1.set_title('Original Pose')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot for optimized pose in green
        x, y, z = optimized_pose_np[num, :, 0], optimized_pose_np[num, :, 2], -optimized_pose_np[num, :, 1]
        ax2.scatter(x, y, z, color='green')
        for pair in pairs:
            ax2.plot([x[pair[0]], x[pair[1]]], [y[pair[0]], y[pair[1]]], [z[pair[0]], z[pair[1]]], color='green')

        ax2.set_title('Optimized Pose')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        return fig


    # Create animation
    ani = animation.FuncAnimation(fig, update_graph, fargs=(original_pose, optimized_pose, pairs),
                                  frames=num_frames, interval=100, blit=False)

    # Save the animation as a video
    import os
    output_dir = 'data/defence'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    filename = f'{output_dir}/{timestamp}_{os.path.basename(file_name)}.mp4'

    print("Timestamp:", timestamp)
    print("File Name:", filename)

    # Assuming ani is your animation object
    ani.save(filename, writer='ffmpeg', fps=10)

    """ x, y, z = original_pose[num, :, 0], original_pose[num, :, 2], -original_pose[num, :, 1]
            ax1.scatter(x, y, z, color='red')
            for pair in pairs:
                ax1.plot([x[pair[0]], x[pair[1]]], [y[pair[0]], y[pair[1]]], [z[pair[0]], z[pair[1]]], color='red')
            ax1.set_title('Original Pose')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')

            # Plot for optimized pose in green
            x, y, z = optimized_pose[num, :, 0], optimized_pose[num, :, 2], -optimized_pose[num, :, 1]
            ax2.scatter(x, y, z, color='green')
            for pair in pairs:
                ax2.plot([x[pair[0]], x[pair[1]]], [y[pair[0]], y[pair[1]]], [z[pair[0]], z[pair[1]]], color='green')
            ax2.set_title('Optimized Pose')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            # Adding text information
            text = f"\nAlpha: {alpha}\nBeta: {beta}\nGamma: {gamma}\nOptimizer: {optimizer_name}\nFile: {file_name}"
            plt.figtext(0.1, 0.8, text, fontsize=10, ha='left')

            return fig,"""


def visualize_optimized_pose_for_defence(optimized_pose, alpha, beta, gamma, file_name):
    num_frames = optimized_pose.shape[0]

    # Define pairs of keypoints to connect
    pairs = [(0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10), (4, 5), (1, 2), (5, 6), (2, 3),
             (8, 11), (8, 14), (11, 12), (14, 15), (12, 13), (15, 16)]

    # Initialize plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=7, azim=50)

    # Function to update each frame in the animation
    def update_graph(num, optimized_pose, pairs):
        ax.clear()

        optimized_pose_np = optimized_pose.detach().cpu().numpy()

        # Plot for optimized pose in green
        x, y, z = optimized_pose_np[num, :, 0], optimized_pose_np[num, :, 2], -optimized_pose_np[num, :, 1]
        ax.scatter(x, y, z, color='green')
        for pair in pairs:
            ax.plot([x[pair[0]], x[pair[1]]], [y[pair[0]], y[pair[1]]], [z[pair[0]], z[pair[1]]], color='green')
        ax.set_title('Optimized Pose')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig

    # Create animation
    ani = animation.FuncAnimation(fig, update_graph, fargs=(optimized_pose, pairs),
                                  frames=num_frames, interval=100, blit=False)

    # Save the animation as a video
    output_dir = 'data/defence'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
    filename = f'{output_dir}/{timestamp}_{os.path.basename(file_name)}.mp4'

    print("Timestamp:", timestamp)
    print("File Name:", filename)

    ani.save(filename, writer='ffmpeg', fps=10)