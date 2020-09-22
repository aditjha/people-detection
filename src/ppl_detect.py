import jetson.inference
import jetson.utils
import rospy
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray


class PeopleDetection:
    def __init__(self):
        self._net = jetson.inference.detectNet("ssd-mobilenet-v2")
        self.img = None
        self.width = None
        self.height = None
        self.need_cam_info = True
        self.camera_model = PinholeCameraModel()
        self.marker_array = MarkerArray()
        self.marker_pub = rospy.Publisher("visualization_markers", MarkerArray, queue_size=500)
        self.camera_info = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.info_callback)

    def get_detections(self, image):
        self.img = jetson.utils.cudaFromNumpy(image)
        self.width = image.shape[1]
        self.height = image.shape[0]
        detections = self._net.Detect(self.img, self.width, self.height)
        print("The inference is happening at " + str(self._net.GetNetworkFPS()) + " FPS")
        return detections, jetson.utils.cudaToNumpy(self.img)

    def get_person_coordinates(self, depth_image, detections):
        coord_list = []
        count = 0
        for det in detections:
            count = count + 1
            if det.ClassID == 1:
                person_center = det.Center
                x, y = person_center
                depth_arr = []
                try:
                    for x in range(int(x) - 2, int(x) + 3):
                        for y in range(int(y) - 2, int(y) + 3):
                            depth_arr.append(depth_image[int(x), int(y)] / 1000.0)
                    depth = np.mean(depth_arr)
                    person_coord = self._get_coord(depth, x, y)
                    self.make_marker(person_coord, count)
                    coord_list.append(person_coord)
                except IndexError:
                    self.marker_pub.publish(self.marker_array)
        self.marker_pub.publish(self.marker_array)
        return coord_list

    def _get_coord(self, person_depth, x, y):
        unit_vector = self.camera_model.projectPixelTo3dRay((x, y))
        normalized_vector = [i / unit_vector[2] for i in unit_vector]
        point_3d = [j * person_depth for j in normalized_vector]
        return point_3d
    
    def make_marker(self, point_3d, count):
        person_marker = Marker()
        person_marker.header.frame_id = "map"
        person_marker.ns = "person"
        person_marker.type = person_marker.SPHERE
        person_marker.action = person_marker.ADD
        person_marker.id = count
        person_marker.pose.position.x = point_3d[0] * -1
        person_marker.pose.position.y = point_3d[2]
        person_marker.pose.position.z = point_3d[1]
        person_marker.pose.orientation.x = 0.0
        person_marker.pose.orientation.y = 0.0
        person_marker.pose.orientation.z = 0.0
        person_marker.pose.orientation.w = 1.0
        person_marker.scale.x = 0.25
        person_marker.scale.y = 0.25
        person_marker.scale.z = 0.25
        person_marker.color.a = 1.0
        person_marker.color.r = 0.0
        person_marker.color.g = 1.0
        person_marker.color.b = 0.0
        person_marker.lifetime = rospy.Duration(1)
        self.marker_array.markers.append(person_marker)

    def info_callback(self, info):
        if self.need_cam_info:
            print("got camera info")
            self.camera_model.fromCameraInfo(info)
            self.need_cam_info = False
