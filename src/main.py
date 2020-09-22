#!/usr/bin/python
"""
This file uses the rgb and depth feed from the topics published by a Intel RealSense camera to detect people and
returns their poses relative to the frame of the camera.
Uses the jetson-inference package found here: https://github.com/dusty-nv/jetson-inference
"""
import rospy
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ppl_detect import PeopleDetection


# function that is called every time there is a new image frame that the ROS subscriber receives
def callback(rgb_image, depth_image, pd_class, bridge, detection_pub):
    """
    callback function that uses rgb frame and depth frame to detect people and prints out their (x,y,z) coordinate
    from the perspective of the camera
    :param rgb_image: rgb frame from realsense camera
    :param depth_image: depth frame which is mapped to same timestamp as rgb frame from realsense camera
    :param pd_class: object instantiated of class PeopleDetection()
    :param bridge: openCV bridge to convert frame to numpy array and vice-versa
    :param detection_pub: rospy publisher node used to publish resulting image with bounding boxes, labels,
    and confidence percentage
    """
    bridge = bridge
    ppl_detect_class = pd_class
    cv_rgb = bridge.imgmsg_to_cv2(rgb_image, "rgba8")
    cv_depth = bridge.imgmsg_to_cv2(depth_image, "passthrough")
    detections, result_img = ppl_detect_class.get_detections(cv_rgb)
    print(type(result_img))
    detection_pub.publish(bridge.cv2_to_imgmsg(result_img, "rgba8"))
    print("detected {:d} objects in image".format(len(detections)))
    coord_results = ppl_detect_class.get_person_coordinates(cv_depth, detections)
    print(coord_results)


def main():
    """
    Initializes rospy subscriber and publisher nodes, object of PeopleDetection class
    Synchronizes rgb and depth frames for matching frames
    Runs indefinitely until user stops
    """
    rospy.init_node('PeopleDetection', anonymous=True)
    print("Running People Detection")

    people_detect = PeopleDetection()
    bridge = CvBridge()
    detection_pub = rospy.Publisher("detected_image", Image)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, people_detect, bridge, detection_pub)
    rospy.spin()


if __name__ == '__main__':
    main()
