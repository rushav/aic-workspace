import rclpy, numpy as np, os, time
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PILImage

rclpy.init()
node = Node('data_collector', parameter_overrides=[
    Parameter('use_sim_time', Parameter.Type.BOOL, True)
])
buf = Buffer()
listener = TransformListener(buf, node)

img_msg = None
cam_info = None

def img_cb(msg):
    global img_msg
    img_msg = msg

def info_cb(msg):
    global cam_info
    cam_info = msg

node.create_subscription(Image, '/center_camera/image', img_cb, 1)
node.create_subscription(CameraInfo, '/center_camera/camera_info', info_cb, 1)

print("Waiting for image and TF data...")
end = time.time() + 15
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)

if img_msg is None:
    print("ERROR: No image received")
    rclpy.shutdown()
    exit(1)

try:
    t = buf.lookup_transform(
        "center_camera/optical",
        "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
        Time()
    )
    p = t.transform.translation
    port_in_cam = np.array([p.x, p.y, p.z])

    fx = cam_info.k[0]
    fy = cam_info.k[4]
    cx = cam_info.k[2]
    cy = cam_info.k[5]

    u = int(fx * port_in_cam[0] / port_in_cam[2] + cx)
    v = int(fy * port_in_cam[1] / port_in_cam[2] + cy)

    print(f"Port in camera frame: x={p.x:.4f} y={p.y:.4f} z={p.z:.4f}")
    print(f"Projected to pixel: u={u}, v={v}")

    arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3)
    img = arr.copy()

    for du in range(-15, 16):
        for dv in range(-15, 16):
            if abs(du) < 2 or abs(dv) < 2:
                pu, pv = u + du, v + dv
                if 0 <= pu < img_msg.width and 0 <= pv < img_msg.height:
                    img[pv, pu] = [255, 0, 0]

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'port_labeled.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PILImage.fromarray(img).save(save_path)
    print(f"Saved to {save_path}")

except Exception as e:
    print(f"TF lookup failed: {e}")
    print("Available frames:")
    print(buf.all_frames_as_string())

node.destroy_node()
rclpy.shutdown()
