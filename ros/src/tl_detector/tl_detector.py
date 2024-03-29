#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3
TRAFFIC_LIGHT_DISTANCE_THRESHOLD = 100.0
CHECK_CAMERA_EVERY_N_FRAMES = 10
STOP_N_WAYPOINTS_BEFORE_TRAFFIC_LIGHT = 30

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        # rospy.loginfo('-----------------------')
        # rospy.loginfo(self.state)
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.previous_distance = 1000000
        self.images_received_counter = 0
        self.is_closing = False

        self.min_dist_to_light = None
        self.closest_light = None

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        # rospy.loginfo('car position: ' + str(self.pose.pose.position.x) + ', ' + str(self.pose.pose.position.y))

        # transform msg to x, y, z position
        self.pose = self.pose.pose.position

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def get_distance(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    def traffic_cb(self, msg):
        self.lights = msg.lights

        if self.pose is None:
            return

        # find the closest traffic light to the car
        min_dist = 1000000.0
        tl =[]
        self.closest_light = None
        # rospy.loginfo('--------------------------------')
        # rospy.loginfo('car position: ' + str(self.pose.x) + ', ' + str(self.pose.y))
        for light in self.lights:
            dist = self.get_distance(self.pose, light.pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                self.closest_light = light.pose.pose.position
            # rospy.loginfo('light position: ' + str(light.pose.pose.position.x) + ', ' + str(light.pose.pose.position.y))
            tl.append(light.pose.pose.position)
        # rospy.loginfo('closest light position: ' + str(self.closest_light.x) + ', ' + str(self.closest_light.y))

        self.min_dist_to_light = min_dist

        # transform msg to x, y, z array
        self.lights = tl

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True

        if not hasattr(self, 'images_received_counter'):
            return

        self.camera_image = msg

        # performs throttling emperically, otherwise PID fails due to high load
        if self.images_received_counter < CHECK_CAMERA_EVERY_N_FRAMES:
            self.images_received_counter = self.images_received_counter + 1
            return
        else:
            self.images_received_counter = 0

        light_wp, state = self.process_traffic_lights()

        if not hasattr(self, 'state'):
            return

        if self.state is None or state is None:
            return

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # rospy.loginfo('------------------')
        # rospy.loginfo(self.state)
        # rospy.loginfo(state)
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        min_dist = 1000000.0
        min_index = -1

        for I in range(1, len(self.waypoints)):
            dist = self.get_distance(pose, self.waypoints[I].pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                min_index = I

        return min_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def get_closest_light(self):

        # since we know the closest light position in world coordinates
        # we need only to find the closest's waypoint index

        if self.closest_light is None:
            return 1000000.0, None, -1

        # find the index of waypoint closest to the closest traffic light
        index = self.get_closest_waypoint(self.closest_light)

        # the stop line is STOP_N_WAYPOINTS_BEFORE_TRAFFIC_LIGHT waypoint steps
        # before the predicted traffic light waypoint index
        return self.min_dist_to_light, self.closest_light, index - STOP_N_WAYPOINTS_BEFORE_TRAFFIC_LIGHT

        # distance = 1000000.0
        # the_light = None
        #
        # for light in self.lights:
        #     d = math.sqrt(math.pow(self.pose.x - light.x, 2) + math.pow(self.pose.y - light.y, 2))
        #     if d < distance:
        #         distance = d
        #         the_light = light
        #
        # return distance, the_light

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        # if(self.pose):
        #     car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
        distance, light, waypoint_index = self.get_closest_light()
        # rospy.loginfo('closest light: ' + str(light.x) + ', ' + str(light.y) +
        #               ', distance: ' + str(distance) + ', wp stop line index: ' + str(waypoint_index))

        if distance > TRAFFIC_LIGHT_DISTANCE_THRESHOLD:
            return -1, TrafficLight.UNKNOWN

        if self.previous_distance - distance > 0.0:
            self.is_closing = True
        else:
            self.is_closing = False

        self.previous_distance = distance

        # rospy.loginfo('is closing to traffic light:' + str(self.is_closing))

        # if light is not None and self.is_closing is True:
        #     rospy.loginfo('light position: ' + str(light.x) + ', '
        #                   + str(light.y) + ', distance: ' + str(distance))

        if light is not None and self.is_closing is True:
            state = self.get_light_state(light)
            # rospy.loginfo('Traffic light recognized:' + str(state))

            rospy.loginfo('Light:' + str(state) +
                            ', Position: ' + str(light.x) + ', ' + str(light.y) +
                            ', Distance: ' + str(distance) +
                            ', Wp Index: ' + str(waypoint_index))
            return waypoint_index, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')