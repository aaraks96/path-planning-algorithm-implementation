import numpy as np
import cv2
import math


        
def check_obstacle_circle(point, dimension, clearance):
    dist_check = dimension + clearance
    center = [190, 20]
    point_x = point[0]
    point_y = point[1]
    dist = np.sqrt((point_x - center[0])**2 + (point_y - center[1])**2)
    if dist <= 15+dist_check+1:
        return True
    else:
        return False


def check_obstacle_ellipse(point, dimension, clearance):
    dist_check = dimension + clearance
    center = [140, 30]
    rx = 15 + dist_check 
    ry = 6 + dist_check +1
    point_x = point[0]
    point_y = point[1]
    dist = (((point_x-center[0])**2)/(rx**2)) + (((point_y-center[1])**2)/(ry**2))
    if dist <= 1:
        return True
    else:
        return False
        
def check_in_square(point, dimension, clearance):
        dist_check = dimension + clearance
        p1 = [50, 37.5]
        p2 = [100, 37.5]
        p3 = [100, 82.5]
        p4 = [50, 82.5]

        point_x = point[0]
        point_y = point[1]

        line1 = point_y +1 - (p1[1] - dist_check)
        line2 = point_x - (p2[0] + dist_check)
        line3 = point_y - (p3[1] + dist_check)
        line4 = point_x - (p4[0] - dist_check)

        flag1 = False
        flag2 = False

        if line1 >= 0 and line3 <= 0:
            flag1 = True
        if line2 <= 0 and line4 >= 0:
            flag2 = True

        if flag1 and flag2 is True:
            return True
        else:
            return False
            
            
def intersect_point(c1, c2):
    det = abs(c1[0] - c2[0])

    x, y = None, None
    if det is not 0:
        x = int(round(abs((c1[1] - c2[1]))/det))
        y = int(round(abs(((c1[0]*c2[1]) - (c2[0]*c1[1])))/det))

    return [x, y]
    
def calc_projection(dist_check,c):
    return dist_check/(math.sin(1.57-math.atan(c)))
    
def new_poly_points(dimension, clearance):
    dist_check = dimension + clearance 
    p1 = [125, 94]
    p2 = [163, 98]
    p3 = [170, 60]
    p4 = [193, 98]
    p5 = [173, 135]
    p6 = [150, 135]

    c1 = np.array(np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1))
    c2 = np.array(np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1))
    c3 = np.array(np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1))
    c4 = np.array(np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1))
    c5 = np.array(np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1))
    c6 = np.array(np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1))

    if dist_check < 1:
        return p1, p2, p3, p4, p5, p6
    else:
        c1[1] = c1[1] - calc_projection(dist_check,c1[0])
        c2[1] = c2[1] - calc_projection(dist_check,c2[0])
        c3[1] = c3[1] - calc_projection(dist_check,c3[0])
        c4[1] = c4[1] + calc_projection(dist_check,c4[0])
        c5[1] = c5[1] + calc_projection(dist_check,c5[0])
        c6[1] = c6[1] + calc_projection(dist_check,c6[0])

        p2 = intersect_point(c1, c2)
        p3 = intersect_point(c2, c3)
        p4 = intersect_point(c3, c4)
        p5 = intersect_point(c4, c5)
        p6 = intersect_point(c5, c6)
        p1 = intersect_point(c6, c1)

        return p1, p2, p3, p4, p5, p6
#Check if init position falls inside obstacle space
def check_in_poly(point, poly_points_updated):
    point_x = point[0]
    point_y = point[1]

    p1 = poly_points_updated[0]
    p2 = poly_points_updated[1]
    p3 = poly_points_updated[2]
    p4 = poly_points_updated[3]
    p5 = poly_points_updated[4]
    p6 = poly_points_updated[5]

    c1 = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    c2 = np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1)
    c3 = np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1)
    c4 = np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1)
    c5 = np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1)
    c6 = np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1)

   
    inner_triangle_1 = np.polyfit([p2[0], p6[0]], [p2[1], p6[1]], 1)
    inner_triangle_2= np.polyfit([p2[0], p4[0]], [p2[1], p4[1]], 1)
    inner_triangle_3 = np.polyfit([p2[0], p5[0]], [p2[1], p5[1]], 1)

    line1 = round(point_y - c1[0] * point_x - (c1[1]))
    line2 = round(point_y - c2[0] * point_x - (c2[1]))
    line3 = round(point_y - c3[0] * point_x - (c3[1]))
    line4 = round(point_y - c4[0] * point_x - (c4[1]))
    line5 = round(point_y - c5[0] * point_x - (c5[1]))
    line6 = round(point_y - c6[0] * point_x - (c6[1]))
    inner_triangle_line_1 = round(point_y - inner_triangle_1[0] * point_x - (inner_triangle_1[1]))
    inner_triangle_line_2 = round(point_y - inner_triangle_2[0] * point_x - (inner_triangle_2[1]))
    inner_triangle_line_3 = round(point_y - inner_triangle_3[0] * point_x - (inner_triangle_3[1]))

    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    

    if line1 >= 0 and inner_triangle_line_1 <= 0 and line6 <= 0:
        flag1 = True
    if line2 >= 0 and line3 >=0 and inner_triangle_line_2 <= 0:
        flag2 = True
    if inner_triangle_line_2 >= 0 and line4 <= 0 and inner_triangle_line_3 <=0:
        flag3 = True
    if inner_triangle_line_3 >= 0 and line5 <= 0 and inner_triangle_line_1 >= 0:
        flag4 = True
    if flag1 or flag2 or flag3 or flag4 is True:
        return True
    else:
        return False

def call_obstacle_checks(point,dimension,clearance, poly_points_updated):
    if check_in_poly(point, poly_points_updated):
        return True
    elif check_in_square(point, dimension, clearance):
        return True
    elif check_obstacle_ellipse(point, dimension, clearance):
        return True
    elif check_obstacle_circle(point, dimension, clearance):
        return True
    
    else:
        return False
    
#plot the obstacle space
def plot_original_workspace():
    img = 255 * np.ones((151, 251, 3), np.uint8)
    poly_points = np.array([[163, 98], [170, 60], [193, 98], [173, 135], [150, 135]],dtype=np.int32)
    poly_points_triangle = np.array([[125, 94], [163, 98], [150, 135]], dtype=np.int32)
    cv2.fillConvexPoly(img, poly_points, 0)
    cv2.fillConvexPoly(img, poly_points_triangle, 0)
    square_points = np.array([[50, 150 - 112.5], [100, 150 - 112.5], [100, 150 - 67.5], [50, 150 - 67.5]],dtype=np.int32)
    cv2.fillConvexPoly(img,square_points, 0)
    cv2.circle(img, (190, 150 - 130), 15, (0, 0, 0), -1)
    cv2.ellipse(img, (140, 150 - 120), (15, 6), 0, 0, 360, 0, -1)
    #resized = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return img
    
#plot the modified obstacle space
def plot_modified_workspace(new_polygon_points,dimension,clearance,img):
    dist_check = dimension+clearance
    image = 255 * np.ones((151, 251, 3), np.uint8)
    stretch = dimension+clearance
    new_square_points = np.array([[50-stretch, 37.5-stretch], [100+stretch, 37.5-stretch], [100+stretch, 82.5+stretch],[50-stretch, 82.5+stretch]], dtype=np.int32)
    cv2.fillConvexPoly(image, new_square_points, 255)
    cv2.circle(image, (190, 150 - 130), 15 + stretch, (255, 0, 0), -1)
    cv2.ellipse(image, (140, 150 - 120), (15 + stretch, 6 + stretch), 0, 0, 360, 255, -1)
    new_polygon = np.array([new_polygon_points[1], new_polygon_points[2], new_polygon_points[3], new_polygon_points[4], new_polygon_points[5]], dtype=np.int32)
    new_polygon_triangle = np.array([new_polygon_points[0],new_polygon_points[1], new_polygon_points[5]], dtype=np.int32)
    cv2.fillConvexPoly(image, new_polygon, 255)
    cv2.fillConvexPoly(image, new_polygon_triangle, 255)
    boundary_points = np.array([[0,0],[0,150],[250,0],[250,150]])
    line1 = np.array([(boundary_points[0][0],boundary_points[0][1]),(boundary_points[0][0],dist_check),(boundary_points[2][0],dist_check),(boundary_points[2][0],boundary_points[2][1])],dtype=np.int32)
    cv2.fillConvexPoly(image, line1, 255)
    line2 = np.array([(boundary_points[0][0], boundary_points[0][1]), (dist_check, boundary_points[0][1]), (dist_check, boundary_points[1][1]), (boundary_points[1][0], boundary_points[1][1])],dtype=np.int32)
    cv2.fillConvexPoly(image,line2,255)
    line3 = np.array([(boundary_points[1][0],boundary_points[1][1]),(boundary_points[1][0],boundary_points[1][1]-dist_check),(boundary_points[3][0],boundary_points[3][1]-dist_check),(boundary_points[3][0],boundary_points[3][1])],dtype = np.int32)
    cv2.fillConvexPoly(image, line3, 255)
    line4 = np.array([(boundary_points[2][0],boundary_points[2][1]),(boundary_points[2][0]-dist_check,boundary_points[2][1]),(boundary_points[3][0]-dist_check,boundary_points[3][1]),(boundary_points[3][0],boundary_points[3][1])],dtype = np.int32)
    cv2.fillConvexPoly(image, line4, 255)
    added_frames = cv2.addWeighted(image,3,img,0.7,0)
    return added_frames





######################################################################################################################################################
def pop_queue_element(queue):
    min = 0
    for elem in range(len(queue)):
        if queue[elem].cost < queue[min].cost:
            min = elem
    
    return queue.pop(min)
        
def find_node(point,queue):
    for elem in queue:
        if(elem.point)==point:
            return queue.index(elem)
        else:
            return None

def moveUp(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_y>0 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x,point_y-1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None
        
def moveDown(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_y<150 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x,point_y+1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None
def moveLeft(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_x>0 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x-1,point_y]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None
def moveRight(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_x<250 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x+1,point_y]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None 
        
def moveDiagUpRight(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x<250 and point_y>0 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x+1,point_y-1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None 
        
def moveDiagUpLeft(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x>0 and point_y>0 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x-1,point_y-1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None 
        
def moveDiagDownRight(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x<250 and point_y<150 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x+1,point_y+1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None
        
def moveDiagDownLeft(point,dimension,clearance,poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x>0 and point_y<150 and not(call_obstacle_checks(point,dimension,clearance, poly_points_updated)):
        new_point = [point_x-1,point_y+1]
        return new_point,base_cost
        #check for obstacles here - return point if false else return None
    else:
        return None,None
        
def generate_node_location(action,current_point,dimension,clearance,poly_points_updated):
    if (action=='up'):
        return moveUp(current_point,dimension,clearance,poly_points_updated)
    if (action=='down'):
        return moveDown(current_point,dimension,clearance,poly_points_updated)
    if (action=='left'):
        return moveLeft(current_point,dimension,clearance,poly_points_updated)
    if (action=='right'):
        return moveRight(current_point,dimension,clearance,poly_points_updated)
    if (action=='diag_up_right'):
        return moveDiagUpRight(current_point,dimension,clearance,poly_points_updated)
    if (action=='diag_up_left'):
        return moveDiagUpLeft(current_point,dimension,clearance,poly_points_updated)
    if (action=='diag_down_right'):
        return moveDiagDownRight(current_point,dimension,clearance,poly_points_updated)
    if (action=='diag_down_left'):
        return moveDiagDownLeft(current_point,dimension,clearance,poly_points_updated)
        
def count_entry_points(point):
    point_x = point[0]
    point_y = point[1]
    count=0
    if point_y>0:
        count+=1
        
    if point_y<150:
        count+=1
        
    if point_x>0:
        count+=1
        
    if point_x<250:
        count+=1
        
    if point_x<250 and point_y>0:
        count+=1
        
    if point_x>0 and point_y>0:
        count+=1
        
    if point_x<250 and point_y<150:
        count+=1
        
    if point_x>0 and point_y<150:
        count+=1
        
    return count

class Node:
    def __init__(self,point):
        self.point = point
        self.cost = math.inf
        self.parent = None
        
 

def show_pixel(image,point):
    image[point[1],point[0]] = [0,255,255]
    return image
    
def trackback(node):
    
    p = list()
    p.append(node.parent)
    parent = node.parent
    if parent == None:
        return p
    
    while parent is not None:
        #print (parent.id)
        p.append(parent)
        parent = parent.parent
    p_rev = list(p)
    return p_rev
    



def Djikstra(start_node_pos,goal_node_pos,poly_points_updated, robot_dimension, robot_clearance):
    
    img = plot_original_workspace()
    image = plot_modified_workspace(poly_points_updated,robot_dimension,robot_clearance,img)

    
    image[start_node_pos[1],start_node_pos[0]] = [0,255,0]
    start_node = Node(start_node_pos)
    start_node.cost = 0
    
    image[goal_node_pos[1],goal_node_pos[0]] = [0,0,255]

    entry_points = count_entry_points(goal_node_pos)
    
    visited = []
    queue = [start_node]
    actions = ["up","down","left","right","diag_up_right","diag_down_right","diag_up_left","diag_down_left"]
    counter = 0

    while queue:
        current_node = pop_queue_element(queue)
        current_point = current_node.point
        visited.append(str(current_point))
        if counter == entry_points:
            return new_node.parent,image
               
        for action in actions:
            new_point, base_cost = generate_node_location(action,current_point,robot_dimension,robot_clearance,poly_points_updated)
            if new_point is not None:
                if new_point == goal_node_pos:
                    if counter<entry_points:
                        counter+=1
                       
                    
                    
                new_node = Node(new_point)
                new_node.parent = current_node
                image = show_pixel(image,current_node.point)
                image[start_node_pos[1],start_node_pos[0]] = [0,255,0]
                image[goal_node_pos[1],goal_node_pos[0]] = [0,0,255]
                resized_new = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Nodes",resized_new)
                cv2.waitKey(1)
                if str(new_point) not in  visited:
                    new_node.cost = base_cost + new_node.parent.cost
                    visited.append(str(new_node.point))
                    queue.append(new_node)
                else:
                    node_exist_index =  find_node(new_point,queue)
                    if node_exist_index is not None:
                        temp_node = queue[node_exist_index]
                        if temp_node.cost > base_cost + new_node.parent.cost:
                           temp_node.cost = base_cost + new_node.parent.cost
                           temp_node.parent = current_node
            else:
                continue
    return None,None

try:
    start_node_x = int(input('Enter start node x postion: '))
    if start_node_x < 0 :
        print("Invalid start node x position, setting x postion to 0")
        start_node_x = 0
    elif start_node_x >250:
        print("Invalid start node x position, setting x postion to 250")
        start_node_x = 250

    start_node_y = int(input('Enter start node y postion: '))
    if start_node_y< 0 :
        print("Invalid start node y position, setting y postion to 0")
        start_node_y = 0
    elif start_node_y >150:
        print("Invalid start node y position, setting y postion to 150")
        start_node_y = 150

    goal_node_x = int(input('Enter goal node x postion: '))
    if goal_node_x < 0 :
        print("Invalid goal node x position, setting x postion to 0")
        goal_node_x = 0
    elif goal_node_x >250:
        print("Invalid goal node x position, setting x postion to 250")
        start_node_x = 250

    goal_node_y = int(input('Enter goal node y postion: '))
    if goal_node_y < 0 :
        print("Invalid goal node y position, setting y postion to 0")
        goal_node_y = 0
    elif goal_node_y >250:
        print("Invalid goal node y position, setting y postion to 250")
        start_node_y = 250
except:
    print("Invalid Input, exiting program")
    exit(0)

start_node_pos = [start_node_x,150-start_node_y]
goal_node_pos = [goal_node_x,150-goal_node_y]


robot_dimension = 0
robot_clearance = 0

poly_points_updated = new_poly_points(robot_dimension,robot_clearance)

s_flag = call_obstacle_checks(start_node_pos,robot_dimension,robot_clearance, poly_points_updated)
g_flag = call_obstacle_checks(goal_node_pos,robot_dimension,robot_clearance, poly_points_updated)

if (s_flag):
    print("Start node in obstacle, program will now exit")
    exit(0)
elif (g_flag):
    print("Goal node in obstacle, program will now exit")
    exit(0)
else:
    result,image = Djikstra(start_node_pos,goal_node_pos,poly_points_updated, robot_dimension, robot_clearance)
    if result is not None:
       nodes_list = trackback(result)
      
       for elem in nodes_list:
            x = elem.point[1]
            y = elem.point[0]
            image[x,y]=[0,255,0]
            image[start_node_pos[1],start_node_pos[0]] = [255,0,0]
            image[goal_node_pos[1],goal_node_pos[0]] = [0,0,255]
            resized_new = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Nodes",resized_new)
            cv2.waitKey(50)
       cv2.waitKey(0)
    
    else:
       print("result is none")

