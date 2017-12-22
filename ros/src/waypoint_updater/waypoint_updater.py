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

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
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
        self.has_wp = False;
        self.car_x = None;
        self.car_y = None;
        self.car_theta = None
        self.car_s = None;
        self.car_v1 = None;
        self.car_v2 = None;
        self.car_v3 = None;
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

        # self.waypoint_index_pub = rospy.Publisher('/current_waypoint', Int32, queue_size=1)

        # TODO: Add other member variables you need below
        # rospy.loginfo("waypoint_updater initialized!")

        self.main_loop()

    def main_loop(self):
        rate = rospy.Rate(10)
        # rospy.loginfo("Looping updater, state is: ")
        # rospy.loginfo(rospy.is_shutdown())
        while not rospy.is_shutdown():

            if (self.base_wp is not None) and (self.cur_pos is not None):
                self.car_x = self.cur_pos.position.x
                self.car_y = self.cur_pos.position.y

                car_s = self.cur_pos.orientation.w

                car_v1 = self.cur_pos.orientation.x
                car_v2 = self.cur_pos.orientation.y
                car_v3 = self.cur_pos.orientation.z

                self.car_theta = 2 * np.arccos(car_s)

                if self.car_theta > np.pi:
                    self.car_theta = -2 * np.pi + self.car_theta

                # if self.has_wp:
                self.next_wp = self.getNextWp()
                # rospy.loginfo([self.car_x,self.car_y])
                # rospy.loginfo("Next WP is: %s",self.next_wp)
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

    def getNextWp(self):
        counter = 0
        x = self.base_wp[counter].pose.pose.position.x
        y = self.base_wp[counter].pose.pose.position.y
        mindist = (self.car_x - x) ** 2 + (self.car_y - y) ** 2

        for i in range(1, self.n_base_wp):
            x = self.base_wp[i].pose.pose.position.x
            y = self.base_wp[i].pose.pose.position.y

            dist = (self.car_x - x) ** 2 + (self.car_y - y) ** 2
            if (dist < mindist):
                mindist = dist
                x = x
                y = y
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
        # msg.header.stamp = rospy.Time
        msg.waypoints = []
        index = next_wp_index

        # rospy.loginfo(self.base_wp[index])

        if self.red_wp == -1:
            self.state = 1
        # rospy.loginfo("STATE IS 1")

        if self.red_wp != -1:
            tf_x = self.base_wp[self.red_wp].pose.pose.position.x
            tl_y = self.base_wp[self.red_wp].pose.pose.position.y
            dist_to_tl = self.dist(self.car_x, tf_x, self.car_y, tl_y)
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

            tf_x = self.base_wp[self.red_wp].pose.pose.position.x
            tl_y = self.base_wp[self.red_wp].pose.pose.position.y

            wp_x = self.base_wp[index].pose.pose.position.x
            wp_y = self.base_wp[index].pose.pose.position.y

            # rospy.loginfo(self.red_wp - index)
            if (self.red_wp + self.n_base_wp > index) and self.red_wp > 0:
                dist_to_tl = self.dist(wp_x, tf_x, wp_y, tl_y)
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
        self.has_wp = True;
        # for index in range(self.n_base_wp):
        #    max_spd = self.base_wp[index].twist.twist.linear.x
        #    rospy.logwarn(max_spd)
        #    rospy.logwarn(index)
        # rospy.loginfo("W_U waypoints loaded!")
        # TODO: Implement
        pass

    def traffic_cb(self, msg):
        self.red_wp = msg.data
        # rospy.loginfo("WP of red light is %s",self.red_wp)
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def dist(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')