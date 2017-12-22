import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, vehicle_mass, wheel_radius,
                 fuel_capacity, accel_limit, decel_limit):
        # TODO: Implement
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.pid_controller = PID(0.4, 0.02, 0.002, -1, 0.30)
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.steer_filter = LowPassFilter(0.0, 1.0)
        self.dbw_enabled = False
        self.throttle = 0
        self.alpha = 0.0
        pass

    def control(self, dbw_enabled_new, cur_lin_vel, cur_ang_vel, des_lin_vel, des_ang_vel):
        self.dbw_enabled = dbw_enabled_new
        if not self.dbw_enabled:
            self.pid_controller.reset()
            return 0., 0., 0.
        # throttle = PID.step()
        steer = self.yaw_controller.get_steering(des_lin_vel, des_ang_vel, cur_lin_vel)
        steer = self.steer_filter.filt(steer)
        newthrottle = self.pid_controller.step(des_lin_vel - cur_lin_vel, 0.02)
        self.throttle = self.throttle * self.alpha + newthrottle * (1.0 - self.alpha)

        # rospy.logwarn(des_lin_vel-cur_lin_vel)
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if des_lin_vel <= 0:
            if self.throttle > 0:
                return 0, self.throttle, steer
            else:
                return 0, -self.throttle, steer

        if (self.throttle > 0):
            return self.throttle, 0, steer
        else:
            # if des_lin_vel-cur_lin_vel > -0.1:
            #	return 0,0,steer
            return 0, -self.throttle, steer

