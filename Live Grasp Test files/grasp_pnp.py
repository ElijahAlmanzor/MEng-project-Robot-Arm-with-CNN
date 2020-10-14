#!/usr/bin/env python
import time
import roslib; roslib.load_manifest('ur_driver')
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from math import pi
import numpy as np
from numpy import linalg
from ur_control.srv import *
from grasp_predict import predict_grasp

import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from grasp_capture_image import take_photo
from grasp_rm_bg import augment

global mat
mat=np.matrix


JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
# Q1 = [1,0,-1.57,0,0,0]
# Q2 = [1.5,0,-1.57,0,0,0]
# Q3 = [1.5,-0.2,-1.57,0,0,0]
Q1 = [3.14,-0.15,-0.157,0,0,-0.157]
#Q2 = [3.14,-0.157,-0.157,0,0,0]
#Q3 = [3.14,-0.157,-0.157,0,0,0.157]
    
client = None





# ****** Coefficients ******


global d1, a2, a3, a7, d4, d5, d6
d1 =  0.089159
a2 = -0.425
a3 = -0.39225
a7 = 0.075
d4 =  0.10915
d5 =  0.09465
d6 =  0.0823

global d, a, alph

d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) #ur5
#d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])#ur10 mm
a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0]) #ur5
#a =mat([0 ,-0.612 ,-0.5723 ,0 ,0 ,0])#ur10 mm
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5
#alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0 ]) # ur10

################# With C
def AH1( n,th,c  ):

  T_a = mat(np.identity(4), copy=False)
  T_a[0,3] = a[0,n-1]
  T_d = mat(np.identity(4), copy=False)
  T_d[2,3] = d[0,n-1]

  Rzt = mat([[cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0],
	         [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0],
	         [0,               0,              1, 0],
	         [0,               0,              0, 1]],copy=False)
      

  Rxa = mat([[1, 0,                 0,                  0],
			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
			 [0, 0,                 0,                  1]],copy=False)

  A_i = T_d * Rzt * T_a * Rxa
	    

  return A_i

def HTrans1(th,c ):  
  A_1=AH1( 1,th,c  )
  A_2=AH1( 2,th,c  )
  A_3=AH1( 3,th,c  )
  A_4=AH1( 4,th,c  )
  A_5=AH1( 5,th,c  )
  A_6=AH1( 6,th,c  )
      
  T_06=A_1*A_2*A_3*A_4*A_5*A_6

  return T_06

########################



# ************************************************** FORWARD KINEMATICS

def AH( n,th  ):

  T_a = mat(np.identity(4), copy=False)
  T_a[0,3] = a[0,n-1]
  T_d = mat(np.identity(4), copy=False)
  T_d[2,3] = d[0,n-1]

  Rzt = mat([[cos(th[n-1]), -sin(th[n-1]), 0 ,0],
	         [sin(th[n-1]),  cos(th[n-1]), 0, 0],
	         [0,               0,              1, 0],
	         [0,               0,              0, 1]],copy=False)
      

  Rxa = mat([[1, 0,                 0,                  0],
			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
			 [0, 0,                 0,                  1]],copy=False)

  A_i = T_d * Rzt * T_a * Rxa
	    

  return A_i

def HTrans(th ):  
  A_1=AH( 1,th  )
  A_2=AH( 2,th  )
  A_3=AH( 3,th  )
  A_4=AH( 4,th  )
  A_5=AH( 5,th  )
  A_6=AH( 6,th  )
      
  T_06=A_1*A_2*A_3*A_4*A_5*A_6

  return T_06

def invKine(desired_pos):# T60
  th = mat(np.zeros((6, 8)))
  P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
  
  # **** theta1 ****
  
  psi = atan2(P_05[2-1,0], P_05[1-1,0])
  phi = acos(d4 /sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
  #The two solutions for theta1 correspond to the shoulder
  #being either left or right
  th[0, 0:4] = pi/2 + psi + phi
  th[0, 4:8] = pi/2 + psi - phi
  th = th.real
  
  # **** theta5 ****
  
  cl = [0, 4]# wrist up or down
  for i in range(0,len(cl)):
	      c = cl[i]
	      T_10 = linalg.inv(AH1(1,th,c))
	      T_16 = T_10 * desired_pos
	      th[4, c:c+2] = + acos((T_16[2,3]-d4)/d6);
	      th[4, c+2:c+4] = - acos((T_16[2,3]-d4)/d6);

  th = th.real
  
  # **** theta6 ****
  # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

  cl = [0, 2, 4, 6]
  for i in range(0,len(cl)):
	      c = cl[i]
	      T_10 = linalg.inv(AH1(1,th,c))
	      T_16 = linalg.inv( T_10 * desired_pos )
	      th[5, c:c+2] = atan2((-T_16[1,2]/sin(th[4, c])),(T_16[0,2]/sin(th[4, c])))
		  
  th = th.real

  # **** theta3 ****
  cl = [0, 2, 4, 6]
  for i in range(0,len(cl)):
	      c = cl[i]
	      T_10 = linalg.inv(AH1(1,th,c))
	      T_65 = AH1( 6,th,c)
	      T_54 = AH1( 5,th,c)
	      T_14 = ( T_10 * desired_pos) * linalg.inv(T_54 * T_65)
	      P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
	      t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2 )/(2 * a2 * a3)) # norm ?
	      th[2, c] = t3.real
	      th[2, c+1] = -t3.real

  # **** theta2 and theta 4 ****

  cl = [0, 1, 2, 3, 4, 5, 6, 7]
  for i in range(0,len(cl)):
	      c = cl[i]
	      T_10 = linalg.inv(AH1( 1,th,c ))
	      T_65 = linalg.inv(AH1( 6,th,c))
	      T_54 = linalg.inv(AH1( 5,th,c))
	      T_14 = (T_10 * desired_pos) * T_65 * T_54
	      P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
	      
	      # theta 2
	      th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3* sin(th[2,c])/linalg.norm(P_13))
	      # theta 4
	      T_32 = linalg.inv(AH1( 3,th,c))
	      T_21 = linalg.inv(AH1( 2,th,c))
	      T_34 = T_32 * T_21 * T_14
	      th[3, c] = atan2(T_34[1,0], T_34[0,0])
  th = th.real

  return th
####################################






def move_repeated(Q11):
    print "Q1n"
    print Q11		
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    try:
        joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position
	print "Current"
	print joints_pos
        d = 2.0
        g.trajectory.points = [JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0))]
        
        g.trajectory.points.append(
                JointTrajectoryPoint(positions=Q11, velocities=[0]*6, time_from_start=rospy.Duration(d)))
            
        client.send_goal(g)
        client.wait_for_result()
    except KeyboardInterrupt:
        client.cancel_goal()
        raise
    except:
        raise

def move_interrupt():
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    try:
        joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position
	temp = HTrans(joints_pos)
	
	print temp[0:2,3]
	print invKine(temp)*180/3.1415

	
        g.trajectory.points = [
            JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
            JointTrajectoryPoint(positions=Q1, velocities=[0]*6, time_from_start=rospy.Duration(2.0)),
            JointTrajectoryPoint(positions=Q2, velocities=[0]*6, time_from_start=rospy.Duration(3.0)),
            JointTrajectoryPoint(positions=Q3, velocities=[0]*6, time_from_start=rospy.Duration(4.0))]

        client.send_goal(g)
        time.sleep(3.0)
        print "Interrupting"
        joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position
        g.trajectory.points = [
            JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0))]
        client.send_goal(g)
        client.wait_for_result()
    #except KeyboardInterrupt:
        #client.cancel_goal()
        raise
    except:
        raise

def gotoxyz(x,y,z):
	print y
	x=x-(+83-96+0.35)/1000
	y=y-(-527+559-0.5)/1000
	z=z-(-298-335-0.56)/1000
	A=[[ 0.08720421, -0.99550347, -0.0369902,   x],
           [-0.99243395, -0.09003705,  0.0834756,   y],
           [-0.08643074,  0.0294309,  -0.99582305,  z],
           [ 0.,          0.,          0.,          1.]]
	temp = invKine(A)
	print temp
	joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position
	print joints_pos
	dis = [0,0,0,0,0,0,0,0]
	print joints_pos[2]
	print temp[0,2]
	print dis[2]
	for i in range(0,8):
		dis[i] =  (joints_pos[0]-temp[0,i])*1+(joints_pos[1]-temp[1,i])*0.2+(joints_pos[2]-temp[2,i])*0.02
	print "dis="
	print dis
	index=np.argmin(dis)
	return np.array(temp[:,2])	

def gotoxyzRPY(x,y,z,R,P,Y):
	print y
	x  = x-(+83-96+0.35)/1000
	y  = y-(-527+559-0.5)/1000
	z  = z-(-298-335-0.56)/1000
	Rm = RPY_trans(R,P,Y)
        A=[[ Rm[0,0], Rm[0,1], Rm[0,2],   x],
           [ Rm[1,0], Rm[1,1], Rm[1,2],   y],
           [ Rm[2,0], Rm[2,1], Rm[2,2],   z],
           [ 0.     ,  0.,          0.,          1.]]
	temp = invKine(A)
	joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position

	dis = [0,0,0,0,0,0,0,0]
	for i in range(0,8):
		dis[i] =  (joints_pos[0]-temp[0,i])*1+(joints_pos[1]-temp[1,i])*0.2+(joints_pos[2]-temp[2,i])*0.02
	index=np.argmin(dis)
        #print np.array(temp[:,2])
        temp2=temp[:,2]
        #temp2=temp2.transpose()
        temp1=np.array((temp[0,2], temp[1,2], temp[2,2], temp[3,2], temp[4,2], temp[5,2]))
	return temp1	



def move_repeated_big(Q11):
		
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    try:
        print 'Move'
        joint_states = rospy.wait_for_message("joint_states", JointState)
        joints_pos = joint_states.position
        jp=np.array(joints_pos)

        print jp
        print Q11
        diff=abs(jp-Q11)
        diff=(diff)
        print diff
        max_diff=max((diff))
        jp=(jp)
        Q11=(Q11)
        print max_diff
        if max_diff>20*pi/180:
            inc=max_diff/(20*pi/180)
            for ij in range(1,int(round(inc))):
                Q12=jp+ij*(Q11-jp)/inc
                print ((Q12))
                move_repeated((Q12))
            move_repeated((Q11))
        else:
            jhu=9
            move_repeated(tuple(Q11)) 
    except KeyboardInterrupt:
        client.cancel_goal()
        raise
    except:
        raise
def RPY_trans(R,P,Y):
    roll=R
    pitch=P
    yaw=Y
    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * math.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta

    return R
class cw:
     def __init__(self, x):
        self.data = x
        
class var:
     def __init__(self, x):
        self.target_width = cw(x)
        

def grip(x):
    #print x
    #print 'Gripper Command'
    rospy.wait_for_service('/rg2_gripper/control_width')
    try:
        grip1 = rospy.ServiceProxy('/rg2_gripper/control_width', RG2)
        grip1(x)
        time.sleep(1)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
   
def main():
    global client
    try:
        rospy.init_node("test_move", anonymous=True, disable_signals=True)
        client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
        print "Waiting for server..."
        client.wait_for_server()
        print "Connected to server"
        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name
        print "This program makes the robot move between the following three poses:"

        print "Please make sure that your robot can move freely between these poses before proceeding!"
             
	inp = raw_input("Continue? y/n: ")[0]
	for test_position in range(1,9):
		if (inp == 'y'):
		    #Go to default position

	  	    '''Q11=gotoxyzRPY(0.6,0.27,-0.05,-3.1415,0,0)
		    move_repeated_big(Q11)
		    r=cw(45) 
		    grip(r)'''

		    
		    Q11=gotoxyzRPY(0.35,-0.05,-0.05,-3.1415,0,0) #Default position
		    move_repeated_big(Q11)
		    r=cw(160) 
		    grip(r)

		    #For the actual grasping
		    #r=cw(0) 
		    #grip(r)
		    #r=cw(45) 
		    #grip(r)
		    #r=cw(0) 
		    #grip(r)

		    #Take photo
		    take_photo(test_position)
		    #remove the bg from the image
		    augment()
		    
		    rgb_image = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image-1.png'
		    d_image = '/home/narly/zed_ur_ws_yf/src/ur_rg2/ur_control/scripts/live_test_image/image_depth-1.png'
		    grasp = predict_grasp(rgb_image, d_image, test_position)
		    print(grasp)

		    #For the x coordinate
		    Ax = np.array([[1,592], [1,250]])
		    x = np.array([[0.63], [0.235]])
		    c = np.dot(linalg.inv(Ax),x)
		    px = grasp[0] #given by the grasp x
		    x_coord = c[0] + c[1]*px
		    #For the y coordinate
		    Ay = np.array([[1,444], [1,15]]) #pixel values of two points
		    y = np.array([[-0.06], [0.27]]) #robot coordinate values of two points
		    d = np.dot(linalg.inv(Ay),y)
		    py = grasp[1] #given by grasp y
		    y_coord = d[0] + d[1]*py - 0.05
		    rot_angle = math.radians(grasp[2])

		    #Move to the grasp location
		    print("rot angle", rot_angle)
		    Q11=gotoxyzRPY(x_coord[0],y_coord[0],-0.1,-3.1415,0,-rot_angle)
		    move_repeated_big(Q11)
		    r=cw(grasp[3]) 
		    grip(r)
		    Q11=gotoxyzRPY(x_coord[0],y_coord[0],-0.2,-3.1415,0,-rot_angle)
		    move_repeated_big(Q11)
		    r=cw(0) 
		    grip(r)
 		    Q11=gotoxyzRPY(x_coord[0],y_coord[0],-0.285,-3.1415,0,-rot_angle)
		    move_repeated_big(Q11)
		    r=cw(45) 
		    grip(r)
 		    Q11=gotoxyzRPY(x_coord[0],y_coord[0],-0.285,-3.1415,0,-rot_angle)
		    move_repeated_big(Q11)
		    Q11=gotoxyzRPY(x_coord[0],y_coord[0],-0.1,-3.1415,0,0)
		    move_repeated_big(Q11)

		    Q11=gotoxyzRPY(0.35,0.0,-0.25,-3.1415,0,0)
		    move_repeated_big(Q11)
                    r=cw(10) 
		    grip(r)

		    '''grasp = predict_grasp('pcd0416r.png', 'pcd0416_depth_image.png')
		    print(grasp)
		    opening_width = grasp[3] + 45
		    xc = grasp[0]/1000
		    yc = grasp[1]/1000
		    rot_angle = math.radians(grasp[2])
		    #open gripper wide
		    r=cw(110) 
		    grip(r)
		    #go to item
		    location=gotoxyzRPY(xc,yc,-0.2,-3.1415,0,rot_angle)
		    move_repeated_big(location) 
		    #lower gripper down
		    pick=gotoxyzRPY(xc,yc,-0.38,-3.1415,0,rot_angle)
		    move_repeated_big(pick) 
		    #grip item
		    r=cw(opening_width)
		    grip(r)
		    #lift object up
		    pick=gotoxyzRPY(xc,yc,-0.2,-3.1415,0,rot_angle)
		    move_repeated_big(pick) '''


		    '''Q11=gotoxyzRPY(0.35,0.15,-0.2,-3.1415,0,0)
		    move_repeated_big(Q11)'''	
		    #Q11=gotoxyzRPY(0.2,0.3,-0.38,-3.1415,0,0)
		    #move_repeated_big(Q11)	
		    #r=cw(45)
		    #grip(r)

		    '''
		    Q11=gotoxyzRPY(0.2,0.3,-0.2,-3.1415,0,0)
		    move_repeated_big(Q11)	

		    Q11=gotoxyzRPY(0.4,0.3,-0.2,-3.1415,0,0)
	 	    move_repeated_big(Q11)	

		    Q11=gotoxyzRPY(0.4,0.3,-0.37,-3.1415,0,0)
		    move_repeated_big(Q11)	

		    r=cw(95)
		    grip(r)


		    r=cw(45)
		    grip(r)

		    Q11=gotoxyzRPY(0.4,0.3,-0.2,-3.1415,0,0)
	 	    move_repeated_big(Q11)	

		    Q11=gotoxyzRPY(0.2,0.3,-0.2,-3.1415,0,0)
		    move_repeated_big(Q11)	

		    Q11=gotoxyzRPY(0.2,0.3,-0.37,-3.1415,0,0)
		    move_repeated_big(Q11)	

		    r=cw(95)
		    grip(r)

		    Q11=gotoxyzRPY(0.2,0.3,-0.2,-3.1415,0,0)
		    move_repeated_big(Q11)	


		    print "Q1="
		    print Q11
		    #print HTrans([[2.716801881790161, -1.1941517035113733, 1.3911328315734863, -1.769663159047262, -1.6051247755633753, -0.5703962484942835]])
		    #move_disordered()
		    #move_interrupt()'''
        else:
            print "Halting program"
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise

if __name__ == '__main__': main()
