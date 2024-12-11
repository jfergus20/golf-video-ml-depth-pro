# take ./data/video.mov
# go frame by frame and run depth_pro.pt

# save all output images to one list or file
# convert list/file into video
import cv2
import os
from PIL import Image
import depth_pro
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats  # Added this import
from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from scipy.interpolate import CubicSpline, splprep, splev, interp1d, UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.ndimage import gaussian_filter1d


DEBUG = True

def get_depth_from_xy_in_png(x, y, depth_heatmap_path):
    """
    Extracts the depth value from the given (x, y) coordinates in the depth heatmap PNG image.
    
    Parameters:
    x (int): The x-coordinate in the heatmap image.
    y (int): The y-coordinate in the heatmap image.
    depth_heatmap_png (np.ndarray): The depth heatmap image data loaded from a PNG file.
    
    Returns:
    float: The depth value at the specified (x, y) coordinates.
    """
    # Ensure the coordinates are within the bounds of the image
    # print("heatmap: ", depth_heatmap_png)
    
    depth_heatmap_png = cv2.imread("./data/midput/" + depth_heatmap_path)

    height, width, _ = depth_heatmap_png.shape

    print("x,y ", x, y)

    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError("Coordinates (x, y) are outside the bounds of the depth heatmap image.")
    
    # Extract the pixel color at the (x, y) coordinates
    pixel_color = depth_heatmap_png[int(y), int(x)]
    
    # Convert the pixel color to a depth value
    # (This will depend on how the depth heatmap was generated and colorized)
    # For example, if the heatmap used the 'turbo' colormap in OpenCV:
    depth_value = (pixel_color[0] / 255) * 10  # Assuming a depth range of 0-10 meters
    
    return depth_value

def run_normal(path=None):
    if DEBUG: print("Video Test Running")
    if path:
        video_path = path
    else:
        video_path = "./data/jamie2copy.mov"
    video = cv2.VideoCapture(video_path)

    if DEBUG: print("Load model and preprocessing transform")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    
    try:
        framecount = 0
        #while video.isOpened():
        while framecount < 2:
            if DEBUG: print("reading frame: ", framecount)
            success, frame = video.read()
            if not success:
                print("failed to read")
                break
            
            if DEBUG: print("Load and preprocess an image.")
            

            imagepath = "./data/midput/frame" + str(framecount) + ".png"

            cv2.imwrite(imagepath, frame) # Save the image
            image, _, f_px = depth_pro.load_rgb(imagepath)
            
            if DEBUG: print("transform")
            image = transform(image)

            if DEBUG: print("Run inference.")
            prediction = model.infer(image, f_px=f_px)
            if DEBUG: print("Inference complete.")
            depth = prediction["depth"]  # Depth in [m].
            print("depth: ", depth[1000,520])
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.

            inverse_depth = 1 / depth
            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )

            # Convert PyTorch tensor to numpy array and proper format for display
            # First detach from computation graph and move to CPU if necessary
            inverse_depth_normalized = inverse_depth_normalized.detach().cpu()
            y = 1000
            x = 520
            #print("idn at ", x, y, inverse_depth_normalized[y,x])

            

            # Convert to numpy and ensure proper range
            visualization = (inverse_depth_normalized.numpy() * 255).astype(np.uint8)
            
            # Apply colormap for better visualization
            depth_colormap = cv2.applyColorMap(visualization, cv2.COLORMAP_TURBO)
            # save this as .png

            output_folder = "./data/midput/"
            # Ensure output folder exists
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            
            # Define input and output paths
            output_path = os.path.join(output_folder, f"frame_{framecount:06d}.png")
            del_path = os.path.join(output_folder, f"frame{framecount}.png")
            if os.path.exists(del_path):
                os.remove(del_path)
            if DEBUG: print(f"Saving processed frame to {output_path}")
            cv2.imwrite(output_path, depth_colormap)

            # Display
            #cv2.imshow('Depth Map', depth_colormap)
            
            # Wait for a key press (1ms) - needed for display to work
            #cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            framecount += 1

        #take file midput and compress it into a video
        """
        

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        """
        input_folder = "./data/midput/"
        output_video = "./data/output.mov"
        # Get list of PNG files
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
        
        if not image_files:
            raise ValueError(f"No PNG files found in {input_folder}")
        
        if DEBUG: print(f"Found {len(image_files)} frames")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
        height, width, layers = first_frame.shape
        
        # Ensure dimensions are even (required for some codecs)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        fps = 30
        # Try different codec combinations that work well with QuickTime
        try:
            # First attempt: H.264 codec
            if output_video.endswith('.mov'):
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            else:
                # Change extension to .mp4 if not .mov
                output_video = os.path.splitext(output_video)[0] + '.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video = cv2.VideoWriter(
                output_video,
                fourcc,
                fps,
                (width, height),
                isColor=True
            )
            
            # Check if VideoWriter was successfully initialized
            if not video.isOpened():
                raise Exception("Failed to initialize video writer with first codec")
                
        except Exception as e:
            if DEBUG: print(f"First codec failed: {e}")
            # Second attempt: Try MPEG-4 codec
            if DEBUG: print("Trying alternative codec...")
            output_video = os.path.splitext(output_video)[0] + '.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(
                output_video,
                fourcc,
                fps,
                (width, height),
                isColor=True
            )
        
        if DEBUG: print(f"Creating video with dimensions: {width}x{height}")
        
        # Write frames to video
        try:
            for image_file in image_files:
                if DEBUG: print(f"Adding frame: {image_file}")
                frame_path = os.path.join(input_folder, image_file)
                frame = cv2.imread(frame_path)
                
                # Resize frame if necessary to match video dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                video.write(frame)
        
        except Exception as e:
            print(f"Error writing frames: {e}")
            raise
        
        finally:
            # Release the video writer
            video.release()
        
        if DEBUG: print(f"Video saved to {output_video}")
        
        # If on macOS, try to make the video compatible with QuickTime using ffmpeg
        try:
            import subprocess
            temp_video = output_video.replace('.mp4', '_temp.mp4')
            os.rename(output_video, temp_video)
            
            # Use ffmpeg to create a QuickTime-compatible video
            cmd = [
                'ffmpeg', '-i', temp_video,
                '-vcodec', 'h264',
                '-acodec', 'aac',
                '-strict', '-2',
                '-pix_fmt', 'yuv420p',
                output_video
            ]
            
            if DEBUG: print("Running ffmpeg conversion...")
            subprocess.run(cmd, check=True)
            os.remove(temp_video)
            if DEBUG: print("FFmpeg conversion completed successfully")
            
        except Exception as e:
            if DEBUG: print(f"FFmpeg conversion failed: {e}")
            # If ffmpeg fails, keep the original video
            if os.path.exists(temp_video):
                os.rename(temp_video, output_video)


        """
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    
        if not image_files:
            raise ValueError(f"No PNG files found in {input_folder}")
        
        if DEBUG: print(f"Found {len(image_files)} frames")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
        height, width, layers = first_frame.shape
        
        # Define the codec and create VideoWriter object
        output_video = "./data/output.mov"
        fps = 10
        if output_video.endswith('.mov'):
            # For .mov files, use 'avc1' codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:
            # Default to mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        video = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )
        
        # Write frames to video
        for image_file in image_files:
            if DEBUG: print(f"Adding frame: {image_file}")
            frame_path = os.path.join(input_folder, image_file)
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        # Release the video writer
        video.release()
        
        if DEBUG: print(f"Video saved to {output_video}")
        """



        """
        image_folder = "./data/midput"
        video_name = "output.mov"
        # Get list of image files (PNG format)
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()  # Sort images to ensure correct order
        
        # Read first image to get dimensions
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
        output_video = "path/to/output"
        fps = 32
        video = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        # Write each image to video
        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            video.write(frame)
        """
    # Release the video writer

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # print("Z: ", Z)
        # print("inv: ", inv_Z)
        cv2.destroyAllWindows()
        if 'video' in locals():
            video.release()

def run_temp():
    if DEBUG: print("running temp")
    input_folder = "./data/midput2"
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
        
    if not frames:
        raise ValueError(f"No PNG files found in {input_folder}")
    if DEBUG: print(f"Found {len(frames)} frames")
    

    # start with no fit, get the process down
    # - predict z at all values != nan
    # - fit 3d nans

    #df = pd.read_csv("./data/xyfit2.csv") # line was fit and smoothed, then nans replaces
    df = pd.read_csv("./data/usethis1interp.csv") # initial xy, with nan values


    xy_list = [[x,y] for x,y in zip(df['x'], df['y'])]
    Z = []
    framecount = 0
    # print(len(xy_list[3:]))
    xy_list = xy_list[3:]
    for x,y in xy_list:
        print("\n*******************************************")
        print(f"frame: {framecount}")
        if np.isnan(x) or np.isnan(y):
            # Append NaN for NaN x,y pairs
            Z.append(np.nan)
        else:
            d = get_depth_from_xy_in_png(x,y, frames[framecount])
            print(f"depth: {d}")
            Z.append(d)
        print("*******************************************")
        framecount += 1

    np.savetxt('./data/z_values1.txt', Z)

    df_new = pd.DataFrame({
        'x': df['x'][3:],
        'y': df['y'][3:],
        'z': Z
    })
    # Save to a new CSV file
    df_new.to_csv("./data/usethisoutput3.csv", index=False)
    print("csv saved")




def predict_z(xy_points, line_params, scalers):
    """
    Predict z values for new x,y points using the fitted line
    """
    scaler_x, scaler_y = scalers
    xy_scaled = scaler_x.transform(xy_points)
    z_scaled = line_params['model'].predict(xy_scaled)
    return scaler_y.inverse_transform(z_scaled)

def fit_3d_line_with_missing_values(df, scaler_x=None, scaler_y=None):
    xy_coords = df[['x', 'y']].values
    z_coords = df[['z']].values
    scaler_x,scaler_y = None,None
    # Create or use scalers
    if scaler_x is None:
        scaler_x = StandardScaler()
        xy_scaled = scaler_x.fit_transform(xy_coords)
    else:
        xy_scaled = scaler_x.transform(xy_coords)
        
    if scaler_y is None:
        scaler_y = StandardScaler()
        z_scaled = scaler_y.fit_transform(z_coords)
    else:
        z_scaled = scaler_y.transform(z_coords)
    
    # Combine scaled coordinates
    scaled_data = np.column_stack([xy_scaled, z_scaled])
    
    # Use KNN imputation for missing values in scaled space
    imputer = KNNImputer(n_neighbors=3)
    completed_data_scaled = imputer.fit_transform(scaled_data)
    
    # Separate back into xy and z
    xy_completed_scaled = completed_data_scaled[:, :2]
    z_completed_scaled = completed_data_scaled[:, 2:]
    
    # Fit line in scaled space
    model = LinearRegression()
    model.fit(xy_completed_scaled, z_completed_scaled)
    
    # Transform data back to original space
    xy_completed = scaler_x.inverse_transform(xy_completed_scaled)
    z_completed = scaler_y.inverse_transform(z_completed_scaled)
    completed_data = np.column_stack([xy_completed, z_completed])
    
    # Package line parameters
    line_params = {
        'slope_x': model.coef_[0][0],  # in scaled space
        'slope_y': model.coef_[0][1],  # in scaled space
        'intercept': model.intercept_[0],  # in scaled space
        'model': model,
        'r2_score': model.score(xy_completed_scaled, z_completed_scaled)
    }
    
    # print(completed_data, line_params, (scaler_x, scaler_y))
    return completed_data, line_params, (scaler_x, scaler_y)

def fillNan():
    df = pd.read_csv('./data/znanfit.csv')
    # Separate x,y coordinates from z coordinates
    # Fit line and get completed data
    completed_data, line_params, scalers = fit_3d_line_with_missing_values(df)
    
    # Make predictions for new points
    #new_points = np.array([[3, 3], [4, 4]])
    #predicted_z = predict_z(new_points, line_params, scalers)
    print(completed_data.shape)
    compdf = pd.DataFrame({
        'x': completed_data[:,0],
        'y': completed_data[:,1],
        'z': completed_data[:,2]
    })
    #compdf = pd.DataFrame(completed_data)
    compdf.to_csv('./data/fillednan.csv', index=False)




def analyze_path_characteristics(df, columns=['x', 'y', 'z']):
    """Analyze path characteristics to determine appropriate thresholds"""
    points = df[columns].values
    
    # Calculate point-to-point distances
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    
    # Calculate angles between consecutive segments
    vectors = np.diff(points, axis=0)
    angles = np.arccos(np.sum(vectors[:-1] * vectors[1:], axis=1) / 
                      (np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)))
    
    return {
        'median_distance': np.median(distances),
        'distance_std': np.std(distances),
        'median_angle': np.median(angles),
        'angle_std': np.std(angles)
    }

def detect_anomalies(df, columns=['x', 'y', 'z'], window_size=7):
    """Detect anomalies using multiple criteria"""
    points = df[columns].values
    anomaly_scores = np.zeros(len(df))
    
    # 1. Distance-based anomalies
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    distances = np.insert(distances, 0, distances[0])
    distance_scores = zscore(distances)
    
    # 2. Angle-based anomalies (sudden direction changes)
    vectors = np.diff(points, axis=0)
    angles = np.zeros(len(df))
    angles[1:-1] = np.arccos(np.sum(vectors[:-1] * vectors[1:], axis=1) / 
                            (np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)))
    angle_scores = zscore(angles)
    
    # 3. Local statistical anomalies
    local_scores = np.zeros(len(df))
    for col in columns:
        rolling_mean = df[col].rolling(window=window_size, center=True).mean()
        rolling_std = df[col].rolling(window=window_size, center=True).std()
        local_scores += np.abs((df[col] - rolling_mean) / rolling_std)
    
    # 4. Curvature anomalies
    curvature = np.zeros(len(df))
    for i in range(1, len(df)-1):
        prev_pt = points[i-1]
        curr_pt = points[i]
        next_pt = points[i+1]
        v1 = curr_pt - prev_pt
        v2 = next_pt - curr_pt
        curvature[i] = np.abs(np.cross(v1, v2)).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2))
    curvature_scores = zscore(curvature)
    
    # Combine scores with weights
    anomaly_scores = (
        1.0 * np.abs(distance_scores) +
        1.0 * np.abs(angle_scores) +
        0.8 * (local_scores / len(columns)) +
        0.5 * np.abs(curvature_scores)
    )
    
    return anomaly_scores

def clean_segment(df, start_idx, end_idx, columns=['x', 'y', 'z']):
    """Clean a segment of the path using adaptive interpolation"""
    buffer = 10  # Points to use before and after for context
    start_fit = max(0, start_idx - buffer)
    end_fit = min(len(df), end_idx + buffer)
    
    df_segment = df.copy()
    num_points = end_idx - start_idx
    
    # Get control points for interpolation
    control_points = np.vstack([
        df[columns].iloc[start_fit:start_idx].values,
        df[columns].iloc[end_idx:end_fit].values
    ])
    
    # Generate interpolated points
    t = np.linspace(0, 1, len(control_points))
    t_new = np.linspace(0, 1, num_points)
    
    # Interpolate each dimension
    for i, col in enumerate(columns):
        cs = CubicSpline(t, control_points[:, i])
        df_segment.loc[start_idx:end_idx-1, col] = cs(t_new)
    
    return df_segment

def clean_path(df, columns=['x', 'y', 'z'], threshold=3.0):
    """Main cleaning function combining multiple techniques"""
    df_clean = df.copy()
    
    # Initial analysis
    path_stats = analyze_path_characteristics(df_clean, columns)
    
    # Detect anomalies
    anomaly_scores = detect_anomalies(df_clean, columns)
    
    # Find continuous segments of anomalies
    is_anomaly = anomaly_scores > threshold
    segments_to_clean = []
    current_segment = []
    
    for i in range(len(df)):
        if is_anomaly[i]:
            if not current_segment:
                current_segment = [i]
            else:
                current_segment.append(i)
        elif current_segment:
            if len(current_segment) >= 2:  # Minimum segment size
                segments_to_clean.append((min(current_segment), max(current_segment) + 1))
            current_segment = []
    
    # Clean identified segments
    for start, end in segments_to_clean:
        df_segment = clean_segment(df_clean, start, end, columns)
        df_clean.loc[start:end-1, columns] = df_segment.loc[start:end-1, columns]
    
    # Final smoothing
    window_size = min(11, len(df) // 5)
    if window_size % 2 == 0:
        window_size += 1
    
    for col in columns:
        df_clean[col] = savgol_filter(df_clean[col], window_size, 3)
    
    return df_clean, segments_to_clean

def drawLineOld():
    # Read the CSV data
    df = pd.read_csv('./data/usethisoutput3.csv')
    
    # Clean the data
    df_cleaned, cleaned_segments = clean_path(df, threshold=3.0)
    df_cleaned = df_cleaned[20:-10]
    cleaned_segments = cleaned_segments[20:-20]
    # Visualization
    fig = plt.figure(figsize=(20, 8))
    #[20:-20]

    
    # Original data
    ax1 = fig.add_subplot(121, projection='3d')
    df_spliced_orig = df.iloc[20:-10]
    ax1.plot3D(df_spliced_orig['x'], df_spliced_orig['z'], df_spliced_orig['y'], 
               'blue', linewidth=2)
    ax1.scatter(df_spliced_orig['x'], df_spliced_orig['z'], df_spliced_orig['y'], 
                c='red', s=1)
    ax1.set_title('Original Data')
    
    # Cleaned data
    ax2 = fig.add_subplot(122, projection='3d')
    df_spliced_clean = df_cleaned.iloc[20:-10]
    ax2.plot3D(df_spliced_clean['x'], df_spliced_clean['z'], df_spliced_clean['y'], 
               'green', linewidth=2)
    ax2.scatter(df_spliced_clean['x'], df_spliced_clean['z'], df_spliced_clean['y'], 
                c='red', s=1)
    
    # Highlight cleaned segments
    for start, end in cleaned_segments:
        if start >= 40 and end <= len(df)-20:
            segment_data = df_cleaned.iloc[start:end]
            ax2.scatter(segment_data['x'], segment_data['z'], segment_data['y'],
                       c='yellow', s=50, alpha=0.5)
    
    ax2.set_title('Cleaned Data (Yellow = Corrected Segments)')
    
    # Common settings
    for ax in [ax1, ax2]:
    # for ax in [ax1]:
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Normalized visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the cleaned data
    df_normalized = df_cleaned.copy()
    for col in ['x', 'y', 'z']:
        df_normalized[col] = (df_cleaned[col] - df_cleaned[col].min()) / (df_cleaned[col].max() - df_cleaned[col].min())
    
    df_spliced = df_normalized.iloc[40:-20]
    
    # ax.plot3D(df_spliced['x'], df_spliced['y'], df_spliced['z'], 'blue', linewidth=2)
    # ax.scatter(df_spliced['x'], df_spliced['y'], df_spliced['z'], c='red', s=1)
    ax.plot3D(df_spliced['x'], df_spliced['z'], df_spliced['y'], 'blue', linewidth=2)
    ax.scatter(df_spliced['x'], df_spliced['z'], df_spliced['y'], c='red', s=1)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('3D Path Visualization (Normalized & Cleaned)')
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def drawBox():
    # get x,y values from csv
    # for each frame
    #   drawbox centered at xy_list[frame_count]
    # inputs: video, xylist
    # outputs: video with boxes drawn
    df = pd.read_csv("./data/twoodsfrontzyinterpedited.csv") # initial xy, with nan values
    xy_list = [[x,y] for x,y in zip(df['x'], df['y'])]

    video_path = "./data/twoangle/TigerWoodsFront.mp4"
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('data', 'tigerwoodszyoutputboxes.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    frame_idx = 0
    box_size = 20
    while True:
        print(f"frame_idx: {frame_idx}")
        success, frame = video.read()
        if not success:
            break
        #print(frame_idx, xy_list[frame_idx])
        # for coords in xy_list[frame_idx]:
        #     if not np.isnan(coords[0]) and not np.isnan(coords[1]):
        #         x,y = int(coords[0]), int(coords[1])
        #         cv2.rectangle(frame, (x - box_size//2, y - box_size//2), (x + box_size//2, y + box_size//2), (0, 255, 0), 2)
                
        for i in range(0, len(xy_list[frame_idx]), 2):
            x = xy_list[frame_idx][i]
            y = xy_list[frame_idx][i+1]
            if not np.isnan(x) and not np.isnan(y):
                #cv2.rectangle(frame, (int(x) - box_size//2, int(y) - box_size//2), (int(x) + box_size//2, int(y) + box_size//2), (0, 255, 0), 2)
                center_x = int(x)
                center_y = int(y)

                # Draw the rectangle
                cv2.rectangle(frame, (center_x - box_size//2, center_y - box_size//2), (center_x + box_size//2, center_y + box_size//2), (0, 255, 0), 2)

                # Draw the crosshairs
                cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
                cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)

            info = [
        
                ("Y: ", y), 
                ("X: ", x),
                ("Frame: ", frame_idx+2),
                # ("Tracker", args["tracker"]),
                # ("Success", "Yes" if success else "No"),
                # ("FPS", "{:.2f}".format(fps.fps())),

            ]
            (H, W) = frame.shape[:2]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 30) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)


        out.write(frame)
        frame_idx += 1

    video.release()
    out.release()
    print("released")

def interpolate_coordinates():
    
    # Load the CSV file into a DataFrame
    input_file = "./data/usethis1edited.csv"
    output_file = "./data/usethis1interp.csv"
    df = pd.read_csv(input_file)
    
    # Interpolate missing x values
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    
    # Interpolate missing y values
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
    
    # Fill remaining NaN values with 0
    df = df.fillna(0)
    
    # Write the interpolated data to a new CSV file
    df.to_csv(output_file, index=False)


##############################




# import pandas as pd
# import numpy as np
# from scipy import stats
# from scipy.signal import savgol_filter
# from scipy.interpolate import splprep, splev
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d

def apply_kalman_smoothing(data):
    """Apply Kalman filter smoothing to a series of points."""
    # Convert input to numpy array if it isn't already
    data = np.array(data)
    
    # Initialize Kalman filter parameters
    Q = 1e-5  # Process variance
    R = 0.1   # Measurement variance
    
    # Allocate space for filtered data
    n = len(data)
    filtered_data = np.zeros(n)
    P = np.zeros(n)  # Error covariance
    
    # Initial guesses
    filtered_data[0] = data[0]
    P[0] = 1.0
    
    # Forward pass
    for i in range(1, n):
        P_pred = P[i-1] + Q
        K = P_pred / (P_pred + R)
        filtered_data[i] = filtered_data[i-1] + K * (data[i] - filtered_data[i-1])
        P[i] = (1 - K) * P_pred
    
    return filtered_data

def smooth_spline(x, y, z, smoothing_factor=0.1):
    """Apply B-spline smoothing to 3D points."""
    # Convert inputs to numpy arrays
    x, y, z = map(np.array, [x, y, z])
    
    points = np.column_stack((x, y, z))
    
    # Calculate the distance between consecutive points
    distances = np.sum(np.diff(points, axis=0)**2, axis=1)**0.5
    u = np.zeros(len(points))
    u[1:] = np.cumsum(distances)
    u /= u[-1]
    
    try:
        # Fit a B-spline
        tck, _ = splprep([x, y, z], u=u, s=smoothing_factor, k=3)
        
        # Generate smooth points
        u_new = np.linspace(0, 1, len(x))
        x_new, y_new, z_new = splev(u_new, tck)
        
        return x_new, y_new, z_new
    except Exception as e:
        print(f"Warning: Spline smoothing failed ({str(e)}), returning original data")
        return x, y, z

def clean_and_smooth_path(df, cleaning_threshold=3.0, smoothing_params=None):
    """Clean and smooth the path using multiple techniques."""
    if smoothing_params is None:
        smoothing_params = {
            'savgol_window': 15,
            'savgol_poly': 3,
            'gaussian_sigma': 2,
            'spline_smoothing': 0.1,
            'kalman': True
        }
    
    # Make a copy of the input data
    df_cleaned = df.copy()
    
    # Step 1: Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df_cleaned[['x', 'y', 'z']]))
    df_cleaned = df_cleaned[(z_scores < cleaning_threshold).all(axis=1)].copy()
    
    # Reset index after removing outliers
    df_cleaned.reset_index(drop=True, inplace=True)
    
    # Step 2: Apply multiple smoothing techniques
    for col in ['x', 'y', 'z']:
        try:
            # Savitzky-Golay filter
            if len(df_cleaned[col]) > smoothing_params['savgol_window']:
                df_cleaned[col] = savgol_filter(df_cleaned[col], 
                                              smoothing_params['savgol_window'], 
                                              smoothing_params['savgol_poly'])
            
            # Gaussian filter
            df_cleaned[col] = gaussian_filter1d(df_cleaned[col], 
                                              smoothing_params['gaussian_sigma'])
            
            # Kalman filter
            if smoothing_params['kalman']:
                df_cleaned[col] = apply_kalman_smoothing(df_cleaned[col].values)
        except Exception as e:
            print(f"Warning: Smoothing failed for column {col}: {str(e)}")
    
    try:
        # Step 3: B-spline smoothing
        x_smooth, y_smooth, z_smooth = smooth_spline(
            df_cleaned['x'].values, 
            df_cleaned['y'].values, 
            df_cleaned['z'].values,
            smoothing_params['spline_smoothing']
        )
        
        df_cleaned['x'] = x_smooth
        df_cleaned['y'] = y_smooth
        df_cleaned['z'] = z_smooth
    except Exception as e:
        print(f"Warning: Spline smoothing failed: {str(e)}")
    
    return df_cleaned

def drawLine2():
    # Read the CSV data
    df = pd.read_csv('./data/usethisoutput3.csv')
    
    # Define smoothing parameters
    smoothing_params = {
        'savgol_window': 5,      # Must be odd number
        'savgol_poly': 3,         # Polynomial order
        'gaussian_sigma': 2,       # Gaussian smoothing factor
        'spline_smoothing': 0.1,   # B-spline smoothing factor
        'kalman': True            # Whether to apply Kalman filtering
    }
    
    # Clean and smooth the data
    df_cleaned = clean_and_smooth_path(df, 
                                     cleaning_threshold=3.0, 
                                     smoothing_params=smoothing_params)
    
    # Trim the data
    #df_cleaned = df_cleaned[20:-10]
    #df_original = df[20:-10]
    df_original = df
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 8))
    
    # Original data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot3D(df_original['x'], df_original['y'], df_original['z'],
               'blue', linewidth=2)
    ax1.scatter(df_original['x'], df_original['y'], df_original['z'],
                c='red', s=1)
    ax1.set_title('Original Data')
    
    # Cleaned and smoothed data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot3D(df_cleaned['x'], df_cleaned['y'], df_cleaned['z'],
               'green', linewidth=2)
    ax2.scatter(df_cleaned['x'], df_cleaned['y'], df_cleaned['z'],
                c='red', s=1)
    ax2.set_title('Cleaned and Smoothed Data')
    
    # Common settings
    for ax in [ax1, ax2]:
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Normalized visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the cleaned data
    df_normalized = df_cleaned.copy()
    for col in ['x', 'y', 'z']:
        min_val = df_cleaned[col].min()
        max_val = df_cleaned[col].max()
        if max_val > min_val:  # Avoid division by zero
            df_normalized[col] = (df_cleaned[col] - min_val) / (max_val - min_val)
    
    ax.plot3D(df_normalized['x'], df_normalized['y'], df_normalized['z'],
              'blue', linewidth=2)
    ax.scatter(df_normalized['x'], df_normalized['y'], df_normalized['z'],
               c='red', s=1)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('3D Path Visualization (Normalized & Smoothed)')
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    df_cleaned.to_csv("./data/cleanedoutput1.csv", index=False)
    print("csv saved")


def standard_scale(df):
    normalized_df = df.copy()
    for column in ['x', 'y', 'z']:
        mean_val = df[column].mean()
        std_val = df[column].std()
        normalized_df[column] = (df[column] - mean_val) / std_val
    return normalized_df

def drawLineKeep():
    # Read the CSV data
    df = pd.read_csv('./data/usethisoutput3.csv')
    
    # Define smoothing parameters
    smoothing_params = {
        'savgol_window': 7,
        'savgol_poly': 2,
        'gaussian_sigma': 1.0,
        'spline_smoothing': 0.05,
        'kalman': False
    }
    
    
    # Create and smooth/normalize data
    df_original = df
    df_cleaned = clean_and_smooth_path(df, 
                                     cleaning_threshold=3.0, 
                                     smoothing_params=smoothing_params)
    df_normalized = standard_scale(df)
    df_clean_normal = standard_scale(df_cleaned)
    df_normal_clean = clean_and_smooth_path(df_normalized,
                                 cleaning_threshold=3.0,
                                 smoothing_params=smoothing_params)
    
    # Create multiple orientations
    fig = plt.figure(figsize=(15, 10))
    
    # Original Orientation
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot3D(df_cleaned['x'], df_cleaned['z'], df_cleaned['y'],
               'blue', linewidth=2)
    ax1.scatter(df_cleaned['x'], df_cleaned['z'], df_cleaned['y'],
                c='red', s=1)
    ax1.set_title('cleaned')
    
    # Flipped Z
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot3D(df_cleaned['x'], df_cleaned['z'], df_cleaned['y'],
               'green', linewidth=2)
    ax2.scatter(df_cleaned['x'], df_cleaned['z'], df_cleaned['y'],
                c='red', s=1)
    ax2.set_title('normalized')
    
    # Cycled Axes
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot3D(df_clean_normal['x'], df_clean_normal['z'], df_clean_normal['y'],
               'purple', linewidth=2)
    ax3.scatter(df_clean_normal['x'], df_clean_normal['z'], df_clean_normal['y'],
                c='red', s=1)
    ax3.set_title('normal(clean)')
    
    # Different Order with Flips
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot3D(df_normal_clean['x'], df_normal_clean['z'], df_normal_clean['y'],
               'orange', linewidth=2)
    ax4.scatter(df_normal_clean['x'], df_normal_clean['z'], df_normal_clean['y'],
                c='red', s=1)
    ax4.set_title('clean(normal)')
    
    # Common settings
    for ax in [ax1, ax2, ax3, ax4]:
        ax.invert_xaxis()
        ax.invert_zaxis()
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # After you find the correct orientation, use this template to create
    # your final visualization with the proper orientation:
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # OPTION 1: Change coordinate order
    # ax.plot3D(df_cleaned['z'], df_cleaned['x'], df_cleaned['y'], 
    #           'blue', linewidth=2)
    
    # OPTION 2: Flip specific axes
    # ax.plot3D(df_cleaned['x'], -df_cleaned['y'], df_cleaned['z'], 
    #           'blue', linewidth=2)
    
    # OPTION 3: Both change order and flip
    # ax.plot3D(-df_cleaned['z'], df_cleaned['x'], df_cleaned['y'], 
    #           'blue', linewidth=2)
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Final Oriented Path')
    # ax.set_box_aspect([1,1,1])
    # ax.view_init(elev=20, azim=45)
    # ax.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    df_cleaned.to_csv("./data/cleanedoutput1.csv", index=False)
    print("csv saved")




def normalize_z_to_range(z, target_min=500, target_max=1200):
    """
    Normalizes z values to the target range while preserving relative relationships
    """
    z_std = (z - np.min(z)) / (np.max(z) - np.min(z))
    z_scaled = z_std * (target_max - target_min) + target_min
    return z_scaled

def fit_z_to_circle(x, y, z):
    """
    Fits z values to form a circular/cardioid pattern with y values
    while preserving the general trend of the original z.
    """
    # Calculate the ideal radius for each point
    t = np.linspace(0, 2*np.pi, len(x))
    # Create a cardioid shape (can be modified to pure circle if needed)
    ideal_z = 2 * np.cos(t) * (1 - np.cos(t))
    
    # Scale and shift the ideal z to match the desired range (500-1200)
    ideal_z = normalize_z_to_range(ideal_z, target_min=500, target_max=1200)
    
    # Blend original and ideal z (weight can be adjusted)
    blend_weight = 0.7  # Higher values favor the ideal shape
    new_z = (1 - blend_weight) * normalize_z_to_range(z) + blend_weight * ideal_z
    
    return new_z

def clean_and_normalize_z(df, smoothing_params):
    """
    Cleans, smooths, and normalizes z values while preserving x and y
    """
    df_cleaned = df.copy()
    z = df_cleaned['z'].values
    
    # Apply Savitzky-Golay filter to z
    if smoothing_params['savgol_window'] > 0:
        z = savgol_filter(z, 
                         smoothing_params['savgol_window'], 
                         smoothing_params['savgol_poly'])
    
    # Apply Gaussian smoothing to z
    if smoothing_params['gaussian_sigma'] > 0:
        z = gaussian_filter1d(z, smoothing_params['gaussian_sigma'])
    
    # Fit to circular/cardioid pattern and normalize
    z = fit_z_to_circle(df_cleaned['x'].values, 
                       df_cleaned['y'].values, 
                       z)
    
    df_cleaned['z'] = z
    return df_cleaned


def normalize_lines(df1, df2):
    """
    Normalize two dataframes representing different angles of the same 3D line
    Ensures y values remain consistent across both datasets
    
    Parameters:
    df1, df2 (pd.DataFrame): Input dataframes with x, y, z columns
    
    Returns:
    tuple: Normalized dataframes
    """
    # Verify common y values
    common_y_check = np.allclose(df1['y'], df2['y'], rtol=0.1)
    if not common_y_check:
        print("Warning: Y values are not consistent. Normalization may not preserve original relationship.")
    
    # Z-score normalization for x and z columns
    columns_to_normalize = ['x', 'z']
    
    for df in [df1, df2]:
        for col in columns_to_normalize:
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
    
    # Keep original y values
    df1['y_normalized'] = df1['y']
    df2['y_normalized'] = df2['y']
    
    return df1, df2

# Usage example



def align_and_normalize_lines(df1, df2):
    """
    Align and normalize two dataframes with different sizes
    
    Parameters:
    df1, df2 (pd.DataFrame): Input dataframes with x, y, z columns
    
    Returns:
    tuple: Normalized and aligned dataframes
    """
    # Interpolate to match sizes
    original_len1, original_len2 = len(df1), len(df2)
    
    # Create interpolation functions for each column
    cols = ['x', 'y', 'z']
    interpolated_dfs = []
    
    for df in [df1, df2]:
        # Create normalized indices
        orig_indices = np.linspace(0, 1, len(df))
        new_indices = np.linspace(0, 1, min(original_len1, original_len2))
        
        # Interpolate each column
        interpolated_cols = {}
        for col in cols:
            interp_func = interp1d(orig_indices, df[col], kind='linear')
            interpolated_cols[col] = interp_func(new_indices)
        
        interpolated_df = pd.DataFrame(interpolated_cols)
        interpolated_dfs.append(interpolated_df)
    
    # Normalize x and z with z-score
    normalized_dfs = []
    for df in interpolated_dfs:
        normalized_df = df.copy()
        for col in ['x', 'z']:
            normalized_df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Normalize y to have consistent span
        y_span = df['y'].max() - df['y'].min()
        normalized_df['y_normalized'] = ((df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())) * y_span
        
        normalized_dfs.append(normalized_df)
    
    return normalized_dfs[0], normalized_dfs[1]




def normalize_combined_lines(df1, df2):
    """
    Combine and normalize two line datasets from different angles
    
    Parameters:
    df1, df2 (pd.DataFrame): Input dataframes with x and y columns
    
    Returns:
    pd.DataFrame: Normalized combined dataframe
    """
    # Interpolate to match sizes
    orig_indices1 = np.linspace(0, 1, len(df1))
    orig_indices2 = np.linspace(0, 1, len(df2))
    new_indices = np.linspace(0, 1, max(len(df1), len(df2)))
    
    # Interpolate x values from first dataset
    x_interp = interp1d(orig_indices1, df1['x'], kind='linear')
    x_normalized = x_interp(new_indices)
    
    # Interpolate y values (ensuring consistency)
    y_interp = interp1d(orig_indices1, df1['y'], kind='linear')
    y_normalized = y_interp(new_indices)
    
    # Interpolate z values from second dataset
    z_interp = interp1d(orig_indices2, df2['x'], kind='linear')
    z_normalized = z_interp(new_indices)
    
    # Create combined dataframe with normalized values
    combined_df = pd.DataFrame({
        'x': (x_normalized - np.mean(x_normalized)) / np.std(x_normalized),
        'y': (y_normalized - np.mean(y_normalized)) / np.std(y_normalized),
        'z': (z_normalized - np.mean(z_normalized)) / np.std(z_normalized)
    })
    
    return combined_df





def drawLine():
    # Read the CSV data
    df = pd.read_csv('./data/two_angle/tigerwoodsxyz.csv')

    # attach x to top of yz
    dftop = pd.read_csv('./data/two_angle/tigerwoodsxyztop.csv')

    # attach x to bottom of yz
    dfmid = pd.read_csv('./data/two_angle/tigerwoodsxyzmid.csv')

    # attach x to middle of yz
    dfend = pd.read_csv('./data/two_angle/tigerwoodsxyzend.csv')







    # Define smoothing parameters
    smoothing_params = {
        'savgol_window': 7,
        'savgol_poly': 2,
        'gaussian_sigma': 1.0,
        'spline_smoothing': 0.05,
        'kalman': False
    }
    
    # Clean and normalize z values
    #df_cleaned = clean_and_normalize_z(df, smoothing_params)

    # normalized_df1, normalized_df2 = normalize_lines(df, df2)
    #normalized_df1, normalized_df2 = align_and_normalize_lines(df, df2)
    # normalized_combined_df = normalize_combined_lines(df, df2)
    # normalized_combined_df.to_csv('./data/normalized_combined.csv', index=False)


    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot3D(dftop['x'], dftop['z'], dftop['y'],
                'blue', linewidth=2)
    ax1.scatter(dftop['x'], dftop['z'], dftop['y'],
                c='red', s=1)
    ax1.set_title('Original (z range: {:.1f} - {:.1f})'.format(
        dftop['z'].min(), dftop['z'].max()))
    
    #ax1.invert_xaxis()
    ax1.invert_zaxis()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_box_aspect([1,1,1])
    ax1.view_init(elev=20, azim=45)
    # Set consistent axis limits
    #ax1.set_ylim(0, 600)  # Z axis

    # Cleaned and Normalized Z
    # ax2 = fig.add_subplot(232, projection='3d')
    # ax2.plot3D(df['x'], df['z'], df['y'],
    #            'green', linewidth=2)
    # ax2.scatter(df['x'], df['z'], df['y'],
    #             c='red', s=1)
    # ax2.set_title('Cleaned & Normalized Z (range: {:.1f} - {:.1f})'.format(
    #     df['z'].min(), df['z'].max()))

    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Z')
    # ax2.set_zlabel('Y')
    # ax2.set_box_aspect([1,1,1])
    # ax2.view_init(elev=20, azim=45)
    # # Set consistent axis limits
    # ax2.set_ylim(400, 1300)  # Z axis
    ax2 = fig.add_subplot(132)
    ax2.plot(dftop['x'], dftop['y'], 'purple', linewidth=2)
    ax2.scatter(dftop['x'], dftop['y'], c='red', s=1)
    ax2.set_title('XY DTL Top')
    ax2.set_aspect('equal')

    #ax2.invert_zaxis()
    ax2.invert_yaxis()

    # YZ Projection (to show circle/cardioid shape)
    ax3 = fig.add_subplot(133)
    ax3.plot(dftop['z'], dftop['y'], 'purple', linewidth=2)
    ax3.scatter(dftop['z'], dftop['y'], c='red', s=1)
    ax3.set_title('ZY Front Top')
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    ax3.set_xlabel('Z')
    ax3.set_ylabel('Y')
    #ax3.set_box_aspect([1,1,1])
    # Set consistent axis limits
    #ax3.set_ylim(1300, 4)  # Z axis
    """
    ax4 = fig.add_subplot(334, projection='3d')
    ax4.plot3D(dfmid['x'], dfmid['z'], dfmid['y'],
                'blue', linewidth=2)
    ax4.scatter(dfmid['x'], dfmid['z'], dfmid['y'],
                c='red', s=1)
    ax4.set_title('Original (z range: {:.1f} - {:.1f})'.format(
        dfmid['z'].min(), dfmid['z'].max()))
    ax4.invert_zaxis()
    ax4.set_xlabel('X')
    ax4.set_ylabel('Z')
    ax4.set_zlabel('Y')
    ax4.set_box_aspect([1,1,1])
    ax4.view_init(elev=20, azim=45)
    # XZ Projection
    # ax4 = fig.add_subplot(234)
    # ax4.plot(df2['z'], df2['x'], 'blue', linewidth=1.5)
    # ax4.scatter(df2['z'], df2['x'], c='red', s=2)
    # ax4.set_title('XZ Projection (Birds Eye)')

    # ax4.invert_xaxis()
    # ax4.set_xlabel('Z')
    # ax4.set_ylabel('X')
    #ax4.set_box_aspect([1,1])

    ax5 = fig.add_subplot(335)
    ax5.plot(dfmid['x'], dfmid['y'], 'blue', linewidth=1.5)
    ax5.scatter(dfmid['x'], dfmid['y'], c='red', s=2)
    ax5.set_title('XY DTL Mid')
    ax5.invert_xaxis()
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    #ax5.set_box_aspect([1,1])
    #Set consistent axis limits
    # ax5.set_ylim(1300, 400)  # Z axis
    
    # ZY Front Projection
    ax6 = fig.add_subplot(336)
    ax6.plot(dfmid['z'], dfmid['y'], 'blue', linewidth=1.5)
    ax6.scatter(dfmid['z'], dfmid['y'], c='red', s=1, zorder=1)
    ax6.set_title('ZY Front Mid')
    ax6.invert_xaxis()
    ax6.set_xlabel('Z')
    ax6.set_ylabel('Y')
    #ax6.set_box_aspect([1,1])


    ax7 = fig.add_subplot(337, projection='3d')
    ax7.plot3D(dfmid['x'], dfmid['z'], dfmid['y'],
                'blue', linewidth=2)
    ax7.scatter(dfmid['x'], dfmid['z'], dfmid['y'],
                c='red', s=1)
    ax7.invert_zaxis()
    ax7.set_xlabel('X')
    ax7.set_ylabel('Z')
    ax7.set_zlabel('Y')
    ax7.set_box_aspect([1,1,1])
    ax7.view_init(elev=20, azim=45)


    ax8 = fig.add_subplot(338)
    ax8.plot(dfend['x'], dfend['y'], 'blue', linewidth=1.5)
    ax8.scatter(dfend['x'], dfend['y'], c='red', s=2)
    ax8.set_title('XY DTL Mid')
    ax8.invert_xaxis()
    ax8.set_xlabel('Z')
    ax8.set_ylabel('X')
    #ax8.set_box_aspect([1,1])

     # ZY Front Projection
    ax9 = fig.add_subplot(339)
    ax9.plot(dfend['z'], dfend['y'], 'blue', linewidth=1.5)
    ax9.scatter(dfend['z'], dfend['y'], c='red', s=1, zorder=1)
    ax9.set_title('ZY Front Mid')
    ax9.invert_xaxis()
    ax9.set_xlabel('Z')
    ax9.set_ylabel('Y')
    #ax9.set_box_aspect([1,1])

"""

    #for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    for ax in [ax1, ax2, ax3]:

        ax.grid(True)



    plt.tight_layout()
    plt.show()
    
    # Save cleaned data
    # df_cleaned.to_csv("./data/cleanedoutput2.csv", index=False)
    print("csv saved")







def video_to_frames(video_path, output_folder, frame_prefix='frame_', frame_format='png'):
    """
    Convert video to a sequence of images.
    
    Args:
        video_path (str): Path to input video file
        output_folder (str): Path to output folder for images
        frame_prefix (str): Prefix for frame filenames
        frame_format (str): Image format to save (png, jpg, etc)
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video with {total_frames} frames at {fps} FPS")
    
    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Generate output filename with leading zeros for proper sorting
        output_path = os.path.join(
            output_folder, 
            f"{frame_prefix}{frame_count:06d}.{frame_format}"
        )
        
        # Save frame
        cv2.imwrite(output_path, frame)
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
            
        frame_count += 1
    
    # Clean up
    cap.release()
    print(f"\nExtracted {frame_count} frames to {output_folder}")



"""
def clean_points(df, columns=['x', 'y', 'z'], window_size=5, threshold=3):
    #Helper function to detect and clean outliers in 3D path data
    df_clean = df.copy()
    
    for col in columns:
        # Calculate rolling mean and standard deviation
        rolling_mean = df_clean[col].rolling(window=window_size, center=True).mean()
        rolling_std = df_clean[col].rolling(window=window_size, center=True).std()
        
        # Identify outliers using z-score
        is_outlier = np.abs((df_clean[col] - rolling_mean) / rolling_std) > threshold
        
        # Replace outliers with interpolated values
        df_clean.loc[is_outlier, col] = np.nan
        df_clean[col] = df_clean[col].interpolate(method='cubic')
        
        # Apply final smoothing
        if len(df_clean) > window_size:
            df_clean[col] = savgol_filter(df_clean[col], 
                                        window_length=window_size, 
                                        polyorder=2)
    
    return df_clean

def drawLine():
    # Read the CSV data
    df = pd.read_csv('./data/z50.csv')
    
    # Clean the data before normalization
    df_cleaned = clean_points(df, window_size=5, threshold=3)
    
    # Create two subplots for comparison
    fig = plt.figure(figsize=(20, 8))
    
    # Original data plot
    ax1 = fig.add_subplot(121, projection='3d')
    df_spliced_orig = df.iloc[40:-20]
    ax1.plot3D(df_spliced_orig['x'], df_spliced_orig['y'], df_spliced_orig['z'], 
               'blue', linewidth=2)
    ax1.scatter(df_spliced_orig['x'], df_spliced_orig['y'], df_spliced_orig['z'], 
                c='red', s=1)
    ax1.set_title('Original Data')
    
    # Cleaned data plot
    ax2 = fig.add_subplot(122, projection='3d')
    df_spliced_clean = df_cleaned.iloc[40:-20]
    ax2.plot3D(df_spliced_clean['x'], df_spliced_clean['y'], df_spliced_clean['z'], 
               'green', linewidth=2)
    ax2.scatter(df_spliced_clean['x'], df_spliced_clean['y'], df_spliced_clean['z'], 
                c='red', s=1)
    ax2.set_title('Cleaned Data')
    
    # Set common properties for both plots
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # If you want to continue with just the cleaned data visualization:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the cleaned data
    df_normalized = df_cleaned.copy()
    df_normalized['x'] = (df_cleaned['x'] - df_cleaned['x'].min()) / (df_cleaned['x'].max() - df_cleaned['x'].min())
    df_normalized['y'] = (df_cleaned['y'] - df_cleaned['y'].min()) / (df_cleaned['y'].max() - df_cleaned['y'].min())
    df_normalized['z'] = (df_cleaned['z'] - df_cleaned['z'].min()) / (df_cleaned['z'].max() - df_cleaned['z'].min())
    
    df_spliced = df_normalized.iloc[40:-20]
    
    # Plot the 3D curve with normalized values
    ax.plot3D(df_spliced['x'], df_spliced['y'], df_spliced['z'], 'blue', linewidth=2)
    ax.scatter(df_spliced['x'], df_spliced['y'], df_spliced['z'], c='red', s=1)
    
    # Set labels
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('3D Path Visualization (Normalized & Cleaned)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

"""

"""
def drawLine():
    # Read the CSV data
    df = pd.read_csv('./data/z50.csv')
    #zhat = savgol_filter(df['z'], len(df['z']), 3) 
    #smooth z before normalizing
    # Normalize the data
    df_normalized = df.copy()
    df_normalized['x'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df_normalized['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    df_normalized['z'] = (df['z'] - df['z'].min()) / (df['z'].max() - df['z'].min())
    # df_spliced = df_normalized.iloc[40:-20]
    df_spliced = df.iloc[40:-20]
    # Create the 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 3D curve with normalized values
    # ax.plot3D(df_normalized['x'], df_normalized['y'], df_normalized['z'], 'blue', linewidth=2)
    ax.plot3D(df_spliced['x'], df_spliced['y'], df_spliced['z'], 'blue', linewidth=2)
    # Add scatter points to better see the path
    # ax.scatter(df_normalized['x'], df_normalized['y'], df_normalized['z'], c='red', s=1)
    ax.scatter(df_spliced['x'], df_spliced['y'], df_spliced['z'], c='red', s=1)
    # Set labels
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('3D Path Visualization (Normalized)')
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    # Make the plot more visually appealing
    ax.grid(True)
    plt.tight_layout()
    # Show the plot
    plt.show()

"""

def run(path=None):
    condition = True
    if condition:
        #run_temp()
        #video_to_frames("./data/output.mov", "./data/midput2")
        #fillNan()
        #linReg()
        # processZ()
        drawLine()
        #drawBox()
        #interpolate_coordinates()
    else:
        run_normal(path)



def main():
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
