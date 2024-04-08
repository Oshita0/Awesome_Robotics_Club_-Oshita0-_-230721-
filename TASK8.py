import math

class Joint:
    def __init__(self, x, y, z, angle):
        self.parent = None
        self.child = None
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle

class FourJointedArm:
    def __init__(self, link_lengths, initial_joint_angles):
        self.link_lengths = link_lengths
        self.joints = [Joint(0, 0, 0, angle) for angle in initial_joint_angles]
        for i in range(len(self.joints) - 1):
            self.joints[i].child = self.joints[i + 1]
            self.joints[i + 1].parent = self.joints[i]

    def set_target(self, x, y, z):
        self.target = (x, y, z)

    def FABRIK(self, max_iterations=100, tolerance=1e-5):
        end_effector = self.get_end_effector()
        target_direction = self.get_target_direction()

        for _ in range(max_iterations):
            # Forward Reachability
            prev_joint = self.joints[0]
            for i in range(1, len(self.joints)):
                curr_joint = self.joints[i]
                curr_joint.x, curr_joint.y, curr_joint.z = self.forward_kinematics(prev_joint, curr_joint.angle)
                prev_joint = curr_joint

            # Backward Reachability
            prev_joint = self.joints[-1]
            for i in range(len(self.joints) - 1, 0, -1):
                curr_joint = self.joints[i]
                dx, dy, dz = self.backward_kinematics(prev_joint, curr_joint, target_direction)
                curr_joint.angle -= dx / self.link_lengths[i - 1]
                curr_joint.angle -= dy / self.link_lengths[i - 1]
                curr_joint.angle -= dz / self.link_lengths[i - 1]
                prev_joint = curr_joint

            # Check Convergence
            if self.is_converged(tolerance):
                break

    def is_converged(self, tolerance):
        end_effector = self.get_end_effector()
        return math.sqrt((end_effector.x - self.target[0])**2 + (end_effector.y - self.target[1])**2 + (end_effector.z - self.target[2])**2) < tolerance

    def get_end_effector(self):
        prev_joint = self.joints[0]
        for i in range(1, len(self.joints)):
            curr_joint = self.joints[i]
            prev_joint = self.forward_kinematics(prev_joint, curr_joint.angle)
        return prev_joint

    def get_target_direction(self):
        target_x, target_y, target_z = self.target
        dx, dy, dz = target_x - self.get_end_effector().x, target_y - self.get_end_effector().y, target_z - self.get_end_effector().z
        return dx, dy, dz

    @staticmethod
    def forward_kinematics(joint, angle):
        x, y, z = joint.x, joint.y, joint.z
        theta = math.radians(angle)
        x += joint.link_length * math.cos(theta)
        y += joint.link_length * math.sin(theta)
        return x, y, z

    @staticmethod
    def backward_kinematics(joint, child_joint, target_direction):
        dx, dy, dz = target_direction
        link_length = joint.link_length
        theta = math.atan2(dy, dx)
        dz -= child_joint.z
        d = math.sqrt(dx**2 + dy**2)
        alpha = math.atan2(dz, d)
        return link_length * math.cos(theta) * math.cos(alpha), link_length * math.sin(theta) * math.cos(alpha), link_length * math.sin(alpha)


if __name__ == "__main__":
    link_lengths = [23, 15, 7]
    initial_joint_angles = [0, 0, 0, 0]
    arm = FourJointedArm(link_lengths, initial_joint_angles)

    target_x, target_y, target_z = 50, 50, 50
    arm.set_target(target_x, target_y, target_z)

    arm.FABRIK()

    print("Target Reachable: ", "Yes" if arm.is_converged(1e-5) else "No")
    print("Optimal Joint Angles: ", [joint.angle for joint in arm.joints])