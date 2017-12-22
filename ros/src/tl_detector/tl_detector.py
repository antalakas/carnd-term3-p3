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
import numpy as np
from random import uniform
import math

STATE_COUNT_THRESHOLD = 6


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
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.red_im_count = 5000
        self.green_im_count = 5000
        self.yellow_im_count = 5000
        self.unknown_im_count = 5000

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # rospy.loginfo(self.waypoints)
        rospy.loginfo("waypoints received!")

    def traffic_cb(self, msg):
        self.lights = msg.lights
        # rospy.loginfo(self.lights)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        # rospy.loginfo("Traffic light image received")
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        if (state == 0):
            state = TrafficLight.RED
        else:
            state = TrafficLight.UNKNOWN

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            # rospy.loginfo("publishing red, wp is %s", light_wp)
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def dist(self, x1, x2, y1, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_closest_waypoint(self, x_val, y_val):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        counter = 0
        if self.waypoints is not None:
            x = self.waypoints.waypoints[0].pose.pose.position.x
            y = self.waypoints.waypoints[0].pose.pose.position.y
            mindist = (x_val - x) ** 2 + (y_val - y) ** 2

            for i in range(1, len(self.waypoints.waypoints)):
                x = self.waypoints.waypoints[i].pose.pose.position.x
                y = self.waypoints.waypoints[i].pose.pose.position.y

                dist = (x_val - x) ** 2 + (y_val - y) ** 2
                if (dist < mindist):
                    mindist = dist
                    x = x
                    y = y
                    counter = i
        else:
            rospy.loginfo("waypoints is None!")
            # rospy.loginfo(self.waypoints)
        return counter

    def get_light_state(self, true_light, dist_to_tl):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        TL_return_value = self.light_classifier.get_classification(cv_image)
        if TL_return_value == TrafficLight.UNKNOWN:
            TL_return_value == 3
        rospy.logwarn("Predict: %s, True: %s", TL_return_value, true_light)

        if False and TL_return_value != true_light:
            if true_light == 0 and (uniform(0, 1) < 0.5) and (dist_to_tl > 100 or dist_to_tl < -1):
                rospy.loginfo("%s", cv_image.shape)
                rospy.loginfo("%s", true_light)
                filename = "/home/student/workspace/carnd-term3-p3/our_dataset/red/" + str(true_light) + "_" + str(
                    self.red_im_count) + ".png"
                cv2.imwrite(filename, cv_image)
                self.red_im_count += 1
            elif true_light == 2 and (uniform(0, 1) < 0.5) and (dist_to_tl > 100 or dist_to_tl < -1):
                rospy.loginfo("%s", cv_image.shape)
                rospy.loginfo("%s", true_light)
                filename = "/home/student/workspace/carnd-term3-p3/our_dataset/green/" + str(true_light) + "_" + str(
                    self.green_im_count) + ".png"
                cv2.imwrite(filename, cv_image)
                self.green_im_count += 1
            elif true_light == 1 and (uniform(0, 1) < 0.5) and (dist_to_tl > 100 or dist_to_tl < -1):
                rospy.loginfo("%s", cv_image.shape)
                rospy.loginfo("%s", true_light)
                filename = "/home/student/workspace/carnd-term3-p3/our_dataset/yellow/" + str(true_light) + "_" + str(
                    self.yellow_im_count) + ".png"
                cv2.imwrite(filename, cv_image)
                self.yellow_im_count += 1
            elif true_light == 3 and (uniform(0, 1) < 0.01):
                rospy.loginfo("%s", cv_image.shape)
                rospy.loginfo("%s", true_light)
                filename = "/home/student/workspace/carnd-term3-p3/our_dataset/unknown/" + str(true_light) + "_" + str(
                    self.unknown_im_count) + ".png"
                cv2.imwrite(filename, cv_image)
                self.unknown_im_count += 1

        # Get classification

        # TL_return_value = 2

        return TL_return_value

    def get_simulator_state(self, stop_pos):
        light_x = self.lights[0].pose.pose.position.x
        light_y = self.lights[0].pose.pose.position.y
        stop_pos_x = stop_pos[0]
        stop_pos_y = stop_pos[1]
        min_dist = self.dist(light_x, stop_pos_x, light_y, stop_pos_y)
        min_value = self.lights[0].state
        minnum = 0
        # print(stop_line_positions)
        # TODO find the closest visible traffic light (if one exists)
        for counter in range(len(self.lights)):
            temp_dist = self.dist(self.lights[counter].pose.pose.position.x, stop_pos_x,
                                  self.lights[counter].pose.pose.position.y, stop_pos_y)
            if temp_dist < min_dist:
                min_dist = temp_dist
                stop_pos_x = self.lights[counter].pose.pose.position.x
                stop_pos_y = self.lights[counter].pose.pose.position.y
                min_value = self.lights[counter].state
                minnum = counter
        return min_value

    def save_image(self, dist_to_tl, stop_pos_x, stop_pos_y):
        state = self.get_simulator_state([stop_pos_x, stop_pos_y])
        state_temp = self.get_light_state(state)
        if dist_to_tl > 100 or dist_to_tl < -2:
            if (uniform(0, 1) < 0.02):
                state_temp = self.get_light_state(1)
        if ((state == 0 or state == 2) and dist_to_tl < 100 and dist_to_tl > -2):
            if state == 0:
                if (uniform(0, 1) < 0.1):
                    state_temp = self.get_light_state(state)
            elif state == 2:
                if (uniform(0, 1) < 0.3):
                    state_temp = self.get_light_state(state)
            else:
                state_temp = self.get_light_state(state)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        if self.waypoints is not None:
            # rospy.loginfo(self.waypoints)

            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']
            if (self.pose):
                car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            vx = self.waypoints.waypoints[(car_position + 1) % len(self.waypoints.waypoints)].pose.pose.position.x - \
                 self.waypoints.waypoints[car_position].pose.pose.position.x
            vy = self.waypoints.waypoints[(car_position + 1) % len(self.waypoints.waypoints)].pose.pose.position.y - \
                 self.waypoints.waypoints[car_position].pose.pose.position.y
            norm_v = np.sqrt(vx * vx + vy * vy)
            vx /= norm_v
            vy /= norm_v
            # rospy.loginfo("Vx = %s",[vx,vy])

            # print(car_position)
            car_x = self.waypoints.waypoints[car_position].pose.pose.position.x
            car_y = self.waypoints.waypoints[car_position].pose.pose.position.y
            stop_pos_x = stop_line_positions[0][0]
            stop_pos_y = stop_line_positions[0][1]
            min_dist = self.dist(car_x, stop_pos_x, car_y, stop_pos_y)
            min_num = 0
            # print(stop_line_positions)
            # TODO find the closest visible traffic light (if one exists)
            for counter in range(len(stop_line_positions)):
                temp_dist = self.dist(car_x, stop_line_positions[counter][0], car_y, stop_line_positions[counter][1])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    stop_pos_x = stop_line_positions[counter][0]
                    stop_pos_y = stop_line_positions[counter][1]
                    min_num = counter

            # rospy.loginfo([stop_pos_x,stop_pos_y])
            light_wp = self.get_closest_waypoint(stop_pos_x, stop_pos_y)

            dx = self.waypoints.waypoints[light_wp].pose.pose.position.x - self.waypoints.waypoints[
                car_position].pose.pose.position.x
            dy = self.waypoints.waypoints[light_wp].pose.pose.position.y - self.waypoints.waypoints[
                car_position].pose.pose.position.y

            dist_to_tl = math.sqrt(self.dist(dx, vx, dy, vy))

            norm_d = np.sqrt(dx * dx + dy * dy)
            if norm_d != 0:
                dx /= norm_d
                dy /= norm_d
            # rospy.loginfo("Dx = %s",[dx,dy])
            dot = vx * dx + vy * dy

            dist_to_tl = dist_to_tl * dot
            # rospy.logwarn("dist_to_tl = %s",dist_to_tl)
            # self.save_image(dist_to_tl,stop_pos_x,stop_pos_y)

            # if light:
            if dot > 0:
                true_state = self.get_simulator_state([stop_pos_x, stop_pos_y])
                state = self.get_light_state(true_state, dist_to_tl)
                # state = true_state

                # print(light_wp, state)
                return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')