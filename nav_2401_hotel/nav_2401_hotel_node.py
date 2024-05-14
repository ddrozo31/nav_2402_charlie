import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid , Odometry , Path
from geometry_msgs.msg import PoseStamped , Twist
from rclpy.qos import QoSProfile

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import heapq

expansion_size = 1 # for the wall, how many values of the matrix will be expanded to the right and left

def euler_from_quaternion(x,y,z,w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def distance(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width) # 2D array reshape from 1D array
    wall = np.where(data == 100) # extract the wall coordinates
    for i in range(-expansion_size,expansion_size+1): #
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:distance(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + distance(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + distance(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = distance(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

class Nav2401HNode(Node):
    def __init__(self):
        super().__init__('nav_2401_hotel_node')
        self.get_logger().info('nav_2401_hotel_node Started')

        self.subscription = self.create_subscription(OccupancyGrid,'/map',self.OccGrid_callback,10)
        self.subscription = self.create_subscription(Odometry,'/diff_cont/odom',self.Odom_callback,10)
        self.subscription = self.create_subscription(PoseStamped,'/goal_pose',self.Goal_Pose_callback,QoSProfile(depth=10))
        self.publisher_visual_path = self.create_publisher(Path, '/visual_path', 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.goal_x = []
        self.goal_y = []

    def OccGrid_callback(self,msg):
        self.get_logger().info('OccupancyGrid Callback')   
        self.resolution = msg.info.resolution
        self.originX = msg.info.origin.position.x
        self.originY = msg.info.origin.position.y
        self.width = msg.info.width
        self.height = msg.info.height
        self.map_data = msg.data # 1D array
        #print(self.resolution,self.originX ,self.originY,self.width,self.height)
        #print(len(self.map_data))

    def Odom_callback(self,msg):
        #self.get_logger().info('Odometry Callback')
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        #print(self.x, self.y, self.yaw)

    def Goal_Pose_callback(self,msg):
        self.get_logger().info('Goal Pose Callback')
        self.goal_x.append(msg.pose.position.x)
        self.goal_y.append(msg.pose.position.y)
        print(self.goal_x, self.goal_y)
                
        #self.follow_path()
        self.get_map()

    def get_map(self):
        data = costmap(self.map_data,self.width,self.height,self.resolution) 
        #print(data)

        column = int((self.x- self.originX)/self.resolution) #x,y 
        row = int((self.y- self.originY)/self.resolution) #x,y 
        
        data[row][column] = 0 #
        data[data < 0] = 1 
        data[data > 5] = 1 

        c_map = cm.get_cmap('rainbow')
        c_map.set_bad('k')
        b = data.copy()
        b[b==0] = np.nan
        fig = plt.imshow(b, interpolation='none', cmap=c_map,origin='lower')
        plt.colorbar(fig)
        plt.show(block=False)
        plt.plot(column,row,'o') # only one set, default # of points



        len_goal_x = len(self.goal_x)
        for i in range(len_goal_x):
            self.goal = (self.goal_x[i],self.goal_y[i])
            
            columnH = int((self.goal[0]- self.originX)/self.resolution)
            rowH = int((self.goal[1]- self.originY)/self.resolution)

            plt.plot(columnH,rowH,'x') # only one set, default # of points
            path = astar(data,(row,column),(rowH,columnH)) 
           
            path = [(p[1]*self.resolution+self.originX,p[0]*self.resolution+self.originY) for p in path] #x,y 
            
            print(path)
            path_points = np.array(path)
            
            plt.plot(path_points,'o') # only one set, default # of points
            plt.show()

            row = rowH
            column = columnH



       
    def follow_path(self):
        self.get_logger().info('follow_path Callback')

        


        a = np.zeros((2))
        a[0]=self.originX
        a[1]=self.originY
        
        b = np.zeros((2))
        b[0]=self.goal[0]
        b[1]=self.goal[1]

        print(distance(a, b))

        fig = plt.figure(figsize=(5,4), dpi=80) # figure size 5x4 inches
        plt.plot(a[0],a[1],'o') # only one set, default # of points
        plt.plot(b[0],b[1],'x')
        plt.xlabel('x') # Labels
        plt.ylabel('y')
        # grid on
        plt.grid()
        plt.show() # presentar


def main(args=None):
    rclpy.init(args=args)
    node = Nav2401HNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()