# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    r = alien.get_width()

    head, tail = alien.get_head_and_tail()

    line_segment = (head, tail)
        
    for line in walls:
        s = ((line[0], line[1]), (line[2], line[3]))
        if segment_distance(line_segment, s) <= r:
            return True
    return False

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    # alien.set_alien_pos((25.5,25.5))
    polygon = [[0, 0], [window[0], 0], [window[0], window[1]], [0, window[1]]]
    center_x, center_y = alien.get_centroid()
    r = alien.get_width()

    width, height = window

    head, tail = alien.get_head_and_tail()

    # is_circle
    if alien.is_circle():
        p1 = [center_x + r, center_y]
        p2 = [center_x - r, center_y]
        p3 = [center_x, center_y + r]
        p4 = [center_x, center_y - r]

        points = [p1, p2, p3, p4]
        for point in points:
            if not is_point_in_polygon(point, polygon):
                return False
        return True
    
    else:
        
        # not a circle
        min_x = min(head[0], tail[0]) - r
        max_x = max(head[0], tail[0]) + r
        min_y = min(head[1], tail[1]) - r
        max_y = max(head[1], tail[1]) + r
        
        if min_x <= 0:
            return False
        if max_x >= width:
            return False
        if min_y <= 0:
            return False
        if max_y >= height:
            return False
        
        return True

# adopted from the following video: https://www.youtube.com/watch?v=RSXM9bgqxJM
def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """

    s1 = [polygon[0], polygon[1]]
    s2 = [polygon[1], polygon[2]]
    s3 = [polygon[2], polygon[3]]
    s4 = [polygon[3], polygon[0]]

    sides = [s1,s2,s3,s4]

    cnt = 0

    for side in sides:
        if (point_segment_distance(point, side)) == 0:
            return True

        (x1, y1), (x2, y2) = side
        if (point[1] < y1) != (point[1] < y2) and point[0] < x1 + ((point[1] - y1)/(y2-y1)) * (x2 - x1):
            cnt += 1

    return cnt % 2 == 1

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall
        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endy), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """ 
    
    center_x, center_y = alien.get_centroid()
    head, tail = alien.get_head_and_tail()
    r = alien.get_width()

    seg_body = (head, tail)

    seg1 = ((center_x, center_y), waypoint)
    seg2 = ((head[0], head[1]), waypoint)
    seg3 = ((tail[0], tail[1]), waypoint)

    for line in walls:
        s = ((line[0], line[1]), (line[2], line[3]))
        if segment_distance(seg_body, s) <= r:
            return True
        if segment_distance(seg1, s) <= r or segment_distance(seg2, s) <= r or segment_distance(seg3, s) <= r:
            return True
    
    # this was sourced from GPT
    # Step 1: Compute the unit vector for the direction of movement
    direction = (waypoint[0] - center_x, waypoint[1] - center_y)
    length = (direction[0]**2 + direction[1]**2) ** 0.5
    if length == 0:
        return False  # No movement
    
    unit_vector = (direction[0] / length, direction[1] / length)

    # Step 2: Move along the path in small steps, checking each position
    steps = int(length)
    for i in range(steps + 1):
        
        # Check if any of the walls intersect with the alien's body at this position
        alien_head = (head[0] + i * unit_vector[0], head[1] + i * unit_vector[1])
        alien_tail = (tail[0] + i * unit_vector[0], tail[1] + i * unit_vector[1])
        seg = (alien_head, alien_tail)

        for wall in walls:
            s = ((wall[0], wall[1]), (wall[2], wall[3]))
            if segment_distance(seg,s) <= r:
                return True

    return False


# Method was converted from this stackoverflow post
# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.

    """
    x,y = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    a = x - x1
    b = y - y1
    c = x2 - x1
    d = y2 - y1

    dot = a* c + b * d
    len_sq = c * c + d * d
    param = -1

    if (len_sq != 0):
        param = dot / len_sq

    xx, yy = None, None

    if (param < 0):
        xx = x1
        yy = y1
    elif (param > 1):
        xx = x2
        yy = y2
    else:
        xx = x1 + param * c
        yy = y1 + param * d

    dx = x - xx
    dy = y - yy

    return math.sqrt(dx * dx + dy * dy)

# This following 3 method was assisted by GeeksForGeeks
# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def check_overlap(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and 
         (q[0] >= min(p[0], r[0])) and 
         (q[1] <= max(p[1], r[1])) and 
         (q[1] >= min(p[1], r[1]))): 
        return True
    return False

def determine_orientation(p, q, r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2
    else: 
        return 0

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """

    p1,q1 = s1
    p2, q2 = s2

    o1 = determine_orientation(p1, q1, p2) 
    o2 = determine_orientation(p1, q1, q2) 
    o3 = determine_orientation(p2, q2, p1) 
    o4 = determine_orientation(p2, q2, q1) 

    if ((o1 != o2) and (o3 != o4)): 
        return True
    if ((o1 == 0) and check_overlap(p1, p2, q1)): 
        return True
    if ((o2 == 0) and check_overlap(p1, q2, q1)): 
        return True
    if ((o3 == 0) and check_overlap(p2, p1, q2)): 
        return True
    if ((o4 == 0) and check_overlap(p2, q1, q2)): 
        return True
  
    return False

# method modified from source: https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments

def segment_distance(s1, s2):

    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        d1 = (point_segment_distance(s1[0], s2))
        d2 = (point_segment_distance(s1[1], s2))
        d3 = (point_segment_distance(s2[0], s1))
        d4 = (point_segment_distance(s2[1], s1))

        min1 = min(d1, d2)
        min2 = min(d3, d4)

        return min(min1, min2)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
