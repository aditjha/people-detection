#!/usr/bin/python
import rospy
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ppl_detect import PeopleDetection


def callback(rgb_image, depth_image, pd_class, bridge, detection_pub):
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
