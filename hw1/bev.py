import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

        # store clicked points from BEV
        self.points = points

    def _euler_to_R(self, theta, phi, gamma, degrees=True):
        if degrees:
            theta, phi, gamma = np.deg2rad([theta, phi, gamma])
        cx, sx = np.cos(theta), np.sin(theta)  # pitch (x)
        cy, sy = np.cos(phi),   np.sin(phi)    # yaw   (y)
        cz, sz = np.cos(gamma), np.sin(gamma)  # roll  (z)
        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]])
        # rotation order: x -> y -> z (pitch, yaw, roll)
        return Rz @ Ry @ Rx

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        # ----- Intrinsics (pinhole, use loaded front image size) -----
        cx = self.width / 2.0
        cy = self.height / 2.0
        f  = (self.width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)  # fx = fy

        # ----- Extrinsics -----
        # World frame: X right, Y up, Z forward
        # Front camera (Camera1): position/orientation
        C_front = np.array([0.0, 1.0, 0.0])
        R_front = np.eye(3)  # orientation (0,0,0)

        # BEV/Top camera (Camera2): position/orientation
        # Use provided Euler angles directly (user expects pitch=0 to face correct direction)
        R_top = self._euler_to_R(theta, phi, gamma, degrees=True)
        C_top = np.array([0.0, 2.5, 0.0])

        new_pixels = []
        if len(self.points) == 0:
            return new_pixels

        # Intersect each BEV ray with ground plane Y=0, then project to front image
        for (u, v) in self.points:
            # Convert BEV pixel to camera ray in BEV camera coordinates.
            # We adopt camera coordinate convention: x right, y up, z forward.
            # Image coordinates: u right, v down -> so y_cam = -(v - cy)
            x_cam = (u - cx) / f
            y_cam = -(v - cy) / f
            z_cam = 1.0
            ray_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

            # direction in world
            d_world = (R_top @ ray_cam)

            # Avoid division by zero (rays parallel to ground plane)
            dy_dir = d_world[1]
            if np.isclose(dy_dir, 0.0):
                continue

            # Ray from BEV cam: P(t) = C_top + t * d_world; intersect with plane Y=0
            t = (0.0 - C_top[1]) / dy_dir
            if t <= 0:  # intersection behind the camera
                continue
            P_world = C_top + t * d_world

            # Transform world -> front camera coordinates
            Pc = (R_front.T @ (P_world - C_front))  # R_front is I, so Pc = P_world - C_front

            Z = Pc[2]
            if Z <= 1e-6:
                continue  # behind the front camera

            u_proj = (f * (Pc[0] / Z)) + cx
            # image v increases downward while world/front-camera y is up -> flip sign
            v_proj = cy - (f * (Pc[1] / Z))

            # clip to image bounds and store as int
            u_i = int(np.clip(np.round(u_proj), 0, self.width - 1))
            v_i = int(np.clip(np.round(v_proj), 0, self.height - 1))
            new_pixels.append([u_i, v_i])

        return new_pixels


    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = 90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)

    projection.show_image(new_pixels)
