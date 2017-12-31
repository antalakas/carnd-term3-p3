#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
BRAKE_DIST = 80
MIN_BRAKE_DIST = 20
BRAKE_VELOCITY_MULT = 0.8
TL_GAP = 8.0
MIN_VELOCITY = 0.25


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.red_light_wp = -1
        self.base_wp = None
        self.cur_pos = None
        self.cur_pos_stamp = None
        self.cur_lin_vel = None
        self.cur_ang_vel = None
        self.car_x = None;
        self.car_y = None;
        self.n_base_wp = 0
        self.near = 0
        self.speed_limit = rospy.get_param('/waypoint_loader/velocity') / 3.6

        self.state = 1  # 0 - STOP, 1 - DRIVE, 2 - BRAKE
        # STOP -> DRIVE, STOP
        # DRIVE -> BRAKE, DRIVE
        # BRAKE -> DRIVE, STOP, BRAKE
        self.red_wp = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        # rospy.loginfo("waypoint_updater initialized!")

        self.main_loop()

    def main_loop(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            if (self.base_wp is not None) and (self.cur_pos is not None):
                self.car_x = self.cur_pos.position.x
                self.car_y = self.cur_pos.position.y

                self.next_wp = self.getNextWp()
                self.publishWaypoints(self.next_wp)
            rate.sleep()

    def pose_cb(self, msg):
        self.cur_pos = msg.pose
        self.cur_pos_stamp = msg.header.stamp
        # TODO: Implement
        pass

    def velocity_cb(self, msg):
        self.cur_lin_vel = msg.twist.linear.x
        self.cur_ang_vel = msg.twist.angular.z
        # rospy.loginfo("Current velocity is: %s",self.cur_lin_vel)

    def get_distance(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    def getNextWp(self):
        counter = 0
        x = self.base_wp[counter].pose.pose.position.x
        y = self.base_wp[counter].pose.pose.position.y
        mindist = 1000000.0

        for i in range(1, self.n_base_wp):
            dist = self.get_distance(self.cur_pos.position, self.base_wp[i].pose.pose.position)
            if (dist < mindist):
                mindist = dist
                counter = i
        next_wp = (counter + 1) % self.n_base_wp

        vx = self.base_wp[next_wp].pose.pose.position.x - x
        vy = self.base_wp[next_wp].pose.pose.position.y - y
        norm_v = np.sqrt(vx * vx + vy * vy)
        if norm_v != 0:
            vx /= norm_v
            vy /= norm_v
        dx = self.car_x - x
        dy = self.car_y - y
        dot = vx * dx + vy * dy
        if dot >= 0:
            return next_wp
        else:
            return counter

    def publishWaypoints(self, next_wp_index):

        msg = Lane()
        msg.waypoints = []
        index = next_wp_index

        if self.red_wp == -1:
            self.state = 1

        if self.red_wp != -1:
            red_position = self.base_wp[self.red_wp].pose.pose.position
            dist_to_tl = self.get_distance(self.cur_pos.position, red_position)
            if dist_to_tl <= BRAKE_DIST:
                if self.state == 1 or self.state == 2:
                    if dist_to_tl < MIN_BRAKE_DIST and self.cur_lin_vel > BRAKE_VELOCITY_MULT * self.speed_limit:
                        self.state = 1
                    # rospy.loginfo("STATE IS 1")
                    else:
                        self.state = 2
                        # rospy.loginfo("STATE IS 2")
                # rospy.loginfo("dist_to_tl, %s",dist_to_tl)
                # rospy.loginfo("TL_GAP, %s",TL_GAP)
                if self.cur_lin_vel < MIN_VELOCITY * 2 and dist_to_tl < TL_GAP:
                    self.state = 0
                # rospy.loginfo("STATE IS 0")

        # rospy.loginfo(self.red_wp - index)
        # rospy.loginfo(index)

        for i in range(LOOKAHEAD_WPS):
            # index of the trailing waypoints
            wp = Waypoint()
            wp.pose.pose.position.x = self.base_wp[index].pose.pose.position.x
            wp.pose.pose.position.y = self.base_wp[index].pose.pose.position.y
            max_spd = self.speed_limit
            max_spd = self.base_wp[index].twist.twist.linear.x

            tf_position = self.base_wp[self.red_wp].pose.pose.position
            wp_position = self.base_wp[index].pose.pose.position

            # rospy.loginfo(self.red_wp - index)
            if (self.red_wp + self.n_base_wp > index) and self.red_wp > 0:
                dist_to_tl = self.get_distance(wp_position, tf_position)
            else:
                dist_to_tl = 0

            if self.state == 2:
                # rospy.loginfo("State is BRAKE, wp dist is %s",dist_to_tl)
                new_vel = min(max(max_spd * (dist_to_tl - TL_GAP * 1.0) / BRAKE_DIST, -0.00), max_spd)
                if new_vel < MIN_VELOCITY and dist_to_tl < TL_GAP * 2 and dist_to_tl > TL_GAP:
                    new_vel = MIN_VELOCITY * 2
                # rospy.loginfo("SET TO MIN_VELOCITY")
                wp.twist.twist.linear.x = new_vel
                # wp.twist.twist.linear.x = 0
                # if wp.twist.twist.linear.x < MIN_VELOCITY:
                if new_vel < MIN_VELOCITY and dist_to_tl < TL_GAP:
                    wp.twist.twist.linear.x = -0.00
                # rospy.loginfo("GOING TO STOP")
                # rospy.loginfo("State is BRAKE, speed is   %s",wp.twist.twist.linear.x)
            else:
                wp.twist.twist.linear.x = -0.00
            if self.state == 1:
                wp.twist.twist.linear.x = max_spd
                if self.cur_lin_vel > self.speed_limit:
                    limit_brake = self.cur_lin_vel - self.speed_limit
                    wp.twist.twist.linear.x = wp.twist.twist.linear.x * max(0.5, (1.0 - 0.75 * limit_brake))
            wp.twist.twist.angular.z = 0
            msg.waypoints.append(wp)
            index = (index + 1) % self.n_base_wp
            # rospy.loginfo(index)

        # rospy.loginfo(msg.waypoints[0].twist.twist.linear.x)
        # rospy.loginfo(self.cur_lin_vel)

        self.final_waypoints_pub.publish(msg)

    def waypoints_cb(self, waypoints):
        self.base_wp = waypoints.waypoints
        self.n_base_wp = len(self.base_wp)
        # TODO: Implement
        pass

    def traffic_cb(self, msg):
        self.red_wp = msg.data
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')