import geopy.distance
import numpy as np
import math
import pandas as pd
from datetime import datetime as dt


class GTFSNetwork:
    """
    Class that stores all related information for a GTFS network
    Processes and generates network and node files from GTFS files
    """
    
    def __init__(self,stop_times_fn,stops_fn,trips_fn, routes_fn, shapes_fn, thresh=0.1, walking_speed=4.5, speed_calc_method='local'):
        '''
        Initializes a GTFSNetwork object
        Cleans GTFS data to prepare for creating a network
        
        Args:
            stop_times_fn (str):    File name of stop_times.txt
            stops_fn (str):         File name of stops.txt
            trips_fn (str):         File name of trips.txt
            routes_fn (str):        File name of routes.txt
            shapes_fn (str):        File name of shapes.txt
            thresh (float):         Maximum distance (mi) to add walking links between nearby stops
            walking_speed (float):  Average walking speed in kph
            speed_calc_method (str):'local' or 'whole' Method by which to calcualte route speed
        '''
        # Store all int versions of string ids
        # int_id    : str_id
        self.int_ids = {}           

        # Load and clean all necessary dataframes
        self.stop_times_df = pd.read_csv(stop_times_fn, low_memory=False, index_col=False)
        self.stop_times_df = self.stop_times_df[['trip_id', 'arrival_time', 'departure_time','stop_id','stop_sequence']]
        
        self.stops_df = pd.read_csv(stops_fn, index_col=False)
        self.stops_df = self.stops_df[['stop_id','stop_lat','stop_lon']]
        
        self.trips_df = pd.read_csv(trips_fn, index_col=False)
        self.trips_df = self.trips_df[['route_id','trip_id','direction_id', 'shape_id']]
        
        self.routes_df = pd.read_csv(routes_fn, index_col=False)
        self.routes_df = self.routes_df[['route_id', 'route_type']]

        self.shapes_df = pd.read_csv(shapes_fn, index_col=False)
        
        # Add shape_dist_traveled if it does not exist
        if 'shape_dist_traveled' not in self.shapes_df.columns:
            self.add_shape_dist_traveled()
        self.shapes_df = self.shapes_df[['shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence','shape_dist_traveled','shape_id']]

        # Convert str to int for trip and stop id's for all dataframes 

        if type(self.stop_times_df.iloc[0]['trip_id']) == str:
            self.stop_times_df['trip_id'] = self.stop_times_df['trip_id'].apply(self.to_int_id)
        if type(self.stop_times_df.iloc[0]['stop_id']) == str:
            self.stop_times_df['stop_id'] = self.stop_times_df['stop_id'].apply(self.to_int_id)

        if type(self.stops_df.iloc[0]['stop_id']) == str:
            self.stops_df['stop_id'] = self.stops_df['stop_id'].apply(self.to_int_id)

        if type(self.trips_df.iloc[0]['trip_id']) == str:
            self.trips_df['trip_id'] = self.trips_df['trip_id'].apply(self.to_int_id)
        if type(self.trips_df.iloc[0]['direction_id']) == str:
            self.trips_df['direction_id'] = self.trips_df['direction_id'].apply(self.to_int_id)

        if type(self.shapes_df.iloc[0]['shape_id'] == str):
            self.shapes_df['shape_id'] = self.shapes_df['shape_id'].apply(self.to_int_id)

        # Remove rows with nan from stop_times
        self.stop_times_df.dropna(subset=["arrival_time","departure_time"], inplace=True)

        # Constants
        self.walking_speed = walking_speed  # kmph
        
        # Generate dicts
        self.generate_id_dicts()
        self.make_unique_shape_pt_ids()

        # Get the bounding box of the area
        north = max(self.stops_df['stop_lat'])
        south = min(self.stops_df['stop_lat'])
        east = max(self.stops_df['stop_lon'])
        west = min(self.stops_df['stop_lon'])
        self.boundaries = [north,south,east,west]
        
        # Create spatial hashing for all stops
        self.thresh = thresh
        self.s_hash = S_Hash(self.boundaries, self.thresh, self.stops_df)

        # Generate spatial hashes for each route shape
        self.route_shape_s_hash = {}   # route id : [s_hash direction 0, s_hash direction 1]
        self.generate_shape_hashes()

        # Generate the route speeds
        if speed_calc_method == 'local':
            self.generate_route_speeds()
        elif speed_calc_method == 'whole':
            self.generate_route_avg_speeds()
        else:
            raise ValueError("Invalid speed_calc_method. Options are 'local' or 'whole'.")

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    # Data Storing/Manipualting Functions                                           #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def to_int_id(self, str_id):
        '''
        Converts string id to int id
        Stores int:str id pairs in self.int_ids

        Args:
            str_id (str):   Any string representing an ID

        Returns:
            int:    Unique int version of the string ID
        '''
        if type(str_id) == str and not str_id.isnumeric():
            int_id = hash(str_id)%100000
        else:
            int_id = int(str_id)
        
        self.int_ids[int_id] = str_id
        
        return int_id
    
    def to_str_id(self, int_id):
        '''
        Converts int id to string id

        Args:
            int_id (int):   Any int ID
        
        Returns:
            str:    Str version of the ID
        '''
        return str(self.int_ids[int_id])

    def make_unique_shape_pt_ids(self):
        '''
        Creates a unique ID for every point in shapes.txt
        IDs do not conflict with stop IDs
        Unique IDs are stored in column 0 of shapes_df
        '''
        max_stop_id = np.max(list(self.stop_to_trip.keys())) + 1
        ids = list(range(max_stop_id, len(self.shapes_df) + max_stop_id))
        self.shapes_df['shape_pt_id'] = ids

        self.shapes_df = self.shapes_df.reindex(columns = ['shape_pt_id', 'shape_pt_lat',
                                                   'shape_pt_lon', 'shape_pt_sequence',
                                                   'shape_dist_traveled','shape_id'])
    
    def add_shape_dist_traveled(self):
        '''
        Adds a column 'shape_dist_traveled' to self.shapes_df
        Distance is calculated based on straight line distance between shape points
        Used when shape_dist_traveled does not exist in shapes.txt
        '''

        # first shape dist is 0
        shape_dist_traveled = [0]
        for i, row in self.shapes_df.iterrows():
            if i == 0:
                continue

            _, prev_lat, prev_lon, prev_sequence = self.shapes_df.iloc[i-1]
            _, curr_lat, curr_lon, curr_sequence = row
            # new shape, start from 0
            if prev_sequence > curr_sequence:
                shape_dist_traveled.append(0)
            # continuing shape, add to total dist
            else:
                dist = geopy.distance.geodesic((prev_lat, prev_lon),(curr_lat, curr_lon)).mi
                tot_dist = dist + shape_dist_traveled[-1]
                shape_dist_traveled.append(tot_dist)
        
        # add shape_dist_traveled to self.shapes_df
        self.shapes_df.insert(len(self.shapes_df.columns), "shape_dist_traveled", shape_dist_traveled, )
            
    def generate_id_dicts(self):
        '''
        Creates all necessary id dicts for quick translation of IDs across files

        route_to_trip   1 : 1 (longest trip for the route)
        trip_to_route   1 : 1

        route_to_shape  1 : 1 (longest shape for the route)
        shape_to_route  1 : 1

        trip_to_shape   1 : 1
        shape_to_trip   1 : 1 (longest trip for the shape)

        trip_to_stop    1 : many
        stop_to_trip    1 : many

        Routes are split by direction:
            route_id = route_id + '_direction_id'
        '''
        self.route_speeds = {}      # route_id  : [speeds]
        self.route_times_dfs = {}   

        self.route_to_trip = {}
        self.trip_to_route = {}

        self.route_to_shape = {}
        self.shape_to_route = {}

        self.trip_to_stop = {}
        self.stop_to_trip = {}

        self.trip_to_shape = {}
        self.shape_to_trip = {}

        # 1: find longest shape for each route, split by direction
        # 1.1: get all shapes for each route
        # 1.2: find the longest shape
        # 1.3: only keep the longest shape
        direction = self.trips_df.iloc[0]['direction_id']
        prev_route_id = self.trips_df.iloc[0]['route_id']
        prev_route_id = str(prev_route_id) + '_' + str(direction)
        
        longest_shape_id = self.trips_df.iloc[0]['shape_id']
        longest_shape_dist = 0
        for _, row in self.trips_df.iterrows():
            route_id, shape_id, direction_id = row['route_id'], row['shape_id'], row['direction_id']
            route_id = str(route_id) + '_' + str(direction_id)

            # on the same route_id -> compare if this shape is longer
            if prev_route_id == route_id:
                # check if this shape is longer
                shape_df = self.shapes_df[self.shapes_df['shape_id'] == shape_id]
                dist = shape_df.iloc[-1]['shape_dist_traveled']
                if dist > longest_shape_dist:
                    longest_shape_dist = dist
                    longest_shape_id = shape_id
            # new route_id -> add the longest route of the previous route_id
            else:
                # route to shape
                self.route_to_shape[prev_route_id] = longest_shape_id
                # shape to route
                self.shape_to_route[longest_shape_id] = prev_route_id
                
                # reset
                prev_route_id = route_id
                longest_shape_id = shape_id
                longest_shape_dist = 0
    
        # 2: find the longest trip for each shape
        self.shape_to_trip = {}
        self.trip_to_shape = {}
        self.route_to_trip = {}
        self.trip_to_route = {}
        for shape_id, route_id in self.shape_to_route.items():
            # get trips that match this shape_id
            trip_ids = self.trips_df.loc[self.trips_df['shape_id'] == shape_id]['trip_id'].to_numpy()
            
            max_sequence = 0
            longest_trip_id = trip_ids[0]
            for trip_id in trip_ids:
                sequence = self.stop_times_df.loc[self.stop_times_df['trip_id'] == trip_id]['stop_sequence'].to_numpy()
                curr_max = np.max(sequence)
                if curr_max > max_sequence:
                    max_sequence = curr_max
                    longest_trip_id = trip_id
            
            # create shape to trip and trip to shape
            # shape to trip
            self.shape_to_trip[shape_id] = longest_trip_id
            # trip to shape
            self.trip_to_shape[longest_trip_id] = shape_id
            
            # route to trip
            self.route_to_trip[route_id] = longest_trip_id

            # trip to route
            self.trip_to_route[longest_trip_id] = route_id

        self.stop_to_trip = {}
        self.trip_to_stop = {}
        # 3: find all stops for all trips
        for trip_id in self.trip_to_shape.keys():
            stop_ids = self.stop_times_df.loc[self.stop_times_df['trip_id'] == trip_id]['stop_id'].to_numpy()
            # trip to stop
            self.trip_to_stop[trip_id] = stop_ids
            # stop to trip
            for stop_id in stop_ids:
                if stop_id not in self.stop_to_trip:
                    self.stop_to_trip[stop_id] = [trip_id]
                else:
                    self.stop_to_trip[stop_id].append(trip_id)

        # generate self.route_times_dfs
        for route_id, trip_id in self.route_to_trip.items():
            route_times_df = self.stop_times_df.loc[self.stop_times_df['trip_id'] == trip_id].copy()
            self.route_times_dfs[route_id] = route_times_df.reset_index(drop=True)


    def translate_id(self, from_id, from_type, to_type):
        '''
        Takes an int id of a given type
        Returns the id of the target type
        Example: use this function to get the route ID that a trip ID belongs to

        Types:
            'route'
            'trip'
            'shape'
            'stop'
        
        Args:
            from_id (any):      Starting ID to be translated
            from_type (str):    Name of the starting ID type
            to_type (str):      Name of the target ID type
        
        Returns:
            Corresponding ID or list of IDs of the target type
        '''

        if from_type == 'route':
            if to_type == 'trip':
                return self.route_to_trip[from_id]

            if to_type == 'shape':
                return self.route_to_shape[from_id]

            if to_type == 'stop':
                # route -> trip -> stop
                trip_id = self.route_to_trip[from_id]
                return self.trip_to_stop[trip_id]

        if from_type == 'trip':
            if to_type == 'route':
                return self.trip_to_route[from_id]

            if to_type == 'shape':
                return self.trip_to_shape[from_id]

            if to_type == 'stop':
                return self.trip_to_stop[from_id]

        if from_type == 'shape':
            if to_type == 'route':
                return self.shape_to_route[from_id]

            if to_type == 'trip':
                return self.shape_to_trip[from_id]

            if to_type == 'stop':
                # shape -> trip -> stop
                trip_id = self.shape_to_trip[from_id]
                return self.trip_to_stop[trip_id]

        if from_type == 'stop':
            if to_type == 'route':
                # stop -> trip -> route
                trip_ids = self.stop_to_trip[from_id]
                routes = []
                for trip_id in trip_ids:
                    route = self.trip_to_route[trip_id]
                    routes.append(route)
                return routes

            if to_type == 'trip':
                return self.stop_to_trip[from_id]

            if to_type == 'shape':
                # stop -> trip -> shape
                trip_ids = self.stop_to_trip[from_id]
                shapes = []
                for trip_id in trip_ids:
                    shape = self.trip_to_shape[trip_id]
                    shapes.append(shape)
                return shapes

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    # Shape Functions                                                               #
    #                                                                               #
    # ----------------------------------------------------------------------------- #

    def generate_shape_hashes(self):
        '''
        Creates S_Hash for each direction of each route_id
        Hash is based on the shape with longest length
        Stored in self.route_shape_s_hash[route_id]
        ''' 

        # Generate shape hash for each route and each direction
        for route_id, shape_id in self.route_to_shape.items():
            # Generate shape hash from longest shape
            shape_df = self.shapes_df[self.shapes_df['shape_id'] == shape_id]

            self.route_shape_s_hash[route_id] = S_Hash(self.boundaries, self.thresh*3, shape_df)

    def generate_route_avg_speeds(self):
        '''
        Calculates the average speed across an entire route for all routes
        Speed is measured in mph
        
        Returns:
            {route_id, average speed}

        '''
        for route in self.route_trips:
            trip_id = self.route_trips[route][0]
            # Get only rows corresponding to 1 trip in stop_times
            stop_times = self.stop_times_df.loc[self.stop_times_df['trip_id']==trip_id]

            # Get total distance traveled
            tot_dist = self.get_route_dist(route)
            
            # Get total time of route in s
            start_time = stop_times.iloc[0]['arrival_time']
            end_time = stop_times.iloc[-1]['arrival_time']
            
            tot_time = self.get_time_diff(start_time, end_time)
            
            # Set the speed for this route in km/h
            self.route_speeds[route] = tot_dist / (tot_time / 3600)
        return self.route_speeds
    
    def generate_route_speeds(self):
        '''
        Calculates the average speed over short edges in the route
        Only edges that are < 7 minutes long are included in the calculation
        All speeds are added to dict route_speeds
        Speed is measured in mph

        Returns:
            {route_id, [(stop_sequence, average speed)]}
        '''
        self.route_speeds = {}
        self.long_edges = {}
        for route_id, route_times_df in self.route_times_dfs.items():
            # start at beginning of the df
            tot_time = 0
            tot_dist = 0
            start_node_id = route_times_df.iloc[0]['stop_id']

            for i in route_times_df.index:
                if i == 0:
                    continue
                from_node_id = route_times_df.iloc[i-1]['stop_id']
                to_node_id = route_times_df.iloc[i]['stop_id']

                from_node_time = route_times_df.iloc[i-1]['arrival_time']
                to_node_time = route_times_df.iloc[i]['arrival_time']

                from_seq = route_times_df.iloc[i-1]['stop_sequence']
                to_seq = route_times_df.iloc[i]['stop_sequence']
                
                edge_time = self.get_time_diff(from_node_time, to_node_time)

                # if link edge is < 7 minutes or is non-consecutive, add to the average
                if edge_time < 7*60 or (to_seq - from_seq != 1):
                    tot_time += edge_time
                
                if edge_time >= 7*60 and (to_seq - from_seq == 1):
                    # add to long edge dict
                    if route_id not in self.long_edges:
                        self.long_edges[route_id] = [{(from_node_id, to_node_id): edge_time}]
                    else:
                        self.long_edges[route_id].append({(from_node_id, to_node_id): edge_time})
                
                # if this is the end of the df, or a long edge was reached,
                # calc the average and add it to the list of speeds
                if edge_time >=7*60 or i == len(route_times_df) - 1:
                    if start_node_id == from_node_id and (to_seq - from_seq == 1):
                        continue
                    tot_dist = self.get_shape_dist(route_id, start_node_id, from_node_id)
                    speed = tot_dist / (tot_time/3600)

                    # start next average at the end node
                    start_node_id = to_node_id

                    sequence = route_times_df.iloc[i]['stop_sequence']
                    
                    if route_id not in self.route_speeds:
                        self.route_speeds[route_id] = [(sequence, speed)]
                    else:
                        self.route_speeds[route_id].append((sequence, speed))

    
    # ----------------------------------------------------------------------------- #
    #                                                                               #
    # Getter Functions                                                               #
    #                                                                               #
    # ----------------------------------------------------------------------------- #

    def get_shape_dist(self, route_id, from_node, to_node):
        '''
        Returns the distance between from_node and to_node on route matching route_id
        Units match distance units of shapes.txt

        Args:
            route_id:   ID of the route
            from_node:  ID of the starting node
            to_node:    ID of the ending node
        
        Returns:
            float: Total distance between starting and ending node  
        '''
        # find the corresponding shape point for first node
        from_lat, from_lon = self.get_node_coords(from_node)

        nearby_pts_from = self.route_shape_s_hash[route_id].get_nearby_stops(from_lat, from_lon)
        shape_pt_0 = float(min(nearby_pts_from, key=nearby_pts_from.get))
        
        # find the shape distance
        dist_0 = self.shapes_df[self.shapes_df['shape_pt_id'] == shape_pt_0]['shape_dist_traveled'].to_numpy()[0]

        # repeat for second node
        to_lat, to_lon = self.get_node_coords(to_node)
        nearby_pts_to = self.route_shape_s_hash[route_id].get_nearby_stops(to_lat, to_lon)
        shape_pt_1 = float(min(nearby_pts_to, key=nearby_pts_to.get))
        
        dist_1 = self.shapes_df[self.shapes_df['shape_pt_id'] == shape_pt_1]['shape_dist_traveled'].to_numpy()[0]

        tot_dist = abs(dist_0 - dist_1)
        return tot_dist
    
    def get_node_coords(self, node_id):
        '''
        Gets the coordinates of the stop matching node_id
        Checks for both stop and shape points

        Args:
            node_id: ID of the target node
        
        Returns:
            [lat, lon]: Coordinates of the target node
        
        '''
        # check stop point
        coords = self.stops_df[self.stops_df['stop_id'] == node_id].to_numpy()
        
        if len(coords) == 0:
            # check shape point
            coords = self.shapes_df[self.shapes_df['shape_pt_id'] == node_id].to_numpy()[0,1:3]
    
        else:
            coords = coords[0, 1:]

        return coords

    def get_time_diff(self, start_time_str, end_time_str):
        '''
        Gets the total seconds between start time and end time

        Args:
            start_time_str (str):   Starting time in foramt HH:MM:SS
            end_time_str (str):     Ending time in format HH:MM:SS
        
        Returns:
            float: Total time elapsed between start and end in seconds
        '''
        try:
            start_time = dt.strptime(start_time_str, "%H:%M:%S")
            end_time = dt.strptime(end_time_str, "%H:%M:%S")
        except(ValueError):
            if '24' in start_time_str[:2]:
                start_time_str = '00' + start_time_str[2:]
            if '24' in end_time_str[:2]:
                end_time_str = '00' + end_time_str[2:]
            start_time = dt.strptime(start_time_str, "%H:%M:%S")
            end_time = dt.strptime(end_time_str, "%H:%M:%S")

        time_diff = (end_time - start_time).total_seconds()
        if time_diff < 0:
            time_diff += 3600 * 24
        return time_diff

    def get_travel_time(self, route_id, from_node_id, to_node_id):
        '''
        Gets the travel time between from_node_id and to_node_id
        Travel time is measured in seconds
        Nodes must be on the same route
        Best used only for consecutive nodes

        Args:
            route_id:   ID of the route shared by starting and ending nodes
            from_node_id:   ID of the starting node
            to_node_id:     ID of the ending node
        
        Returns:
            float: Travel time between nodes measured in seconds

        '''
        # If this route has long edges, check if this is one of them
        if route_id in self.long_edges:
            long_edges = self.long_edges[route_id]
            # Check if the edge is > 7 minutes
            if (from_node_id,to_node_id) in long_edges:
                return long_edges[(from_node_id, to_node_id)]
            elif (to_node_id, from_node_id) in long_edges:
                return long_edges[(to_node_id, from_node_id)]


        route_times_df = self.route_times_dfs[route_id]
        
        # Find the sequence of each stop
        try:
            from_seq = route_times_df.loc[route_times_df['stop_id']==from_node_id]['stop_sequence'].to_numpy()[0]
            to_seq = route_times_df.loc[route_times_df['stop_id']==to_node_id]['stop_sequence'].to_numpy()[0]
        except:
            print('from, to', from_node_id, to_node_id)
        sequence = np.max([from_seq, to_seq])

        speeds = self.route_speeds[route_id]
        speed = 0

        for speed_pair in speeds:
            if sequence <= speed_pair[0]:
                speed = speed_pair[1]
                break
        
        # Ensure no 0 divide
        if speed == 0:
            speed = 10

        travel_time = self.get_shape_dist(route_id, from_node_id, to_node_id) / speed # hours

        return travel_time * 3600 # seconds
    
    def get_geo_dist(self,from_node_id,to_node_id):
        '''
        Calculates the straight line distance between nodes

        Args:
            from_node_id:   ID of the starting node
            to_node_id:     ID of the ending node
        
        Returns:
            float: Straight line distance between nodes measured in miles
        '''
        from_node_coords = self.get_node_coords(int(float(from_node_id)))
        to_node_coords = self.get_node_coords(int(float(to_node_id)))
        return geopy.distance.geodesic(from_node_coords,to_node_coords).mi

    def get_route_dist(self,route_id):
        '''
        Calculates the total distance for the route

        Args:
            route_id: ID of the target route
        
        Returns:
            float: The distance of the route. Measured in the units used in shapes.txt
        '''
        shape_ids = list(self.translate_id(route_id, 'route', 'shape'))
        dists = []
        for shape_id in shape_ids:
            dists.append(self.get_shape_id_dist(shape_id))
        return np.max(dists)
    
    """
    def get_avg_headway(self, route_id):
        '''
        Calculates and returns the average headway for the route matching route_id
        Measured in minutes

        Args:
            route_ID: ID of the target route
        
        Returns:
            float: Average headway of the target route measured in minutes
        '''
        route_id = route_id[:-2] # remove direction_id
        trip_ids = self.trips_df.loc[self.trips_df['route_id'] == route_id]['trip_id'].to_numpy()
        
        # Get stop_times_df where route_id matches
        start_times = []
        for trip_id in trip_ids:
            stop_times = self.stop_times_df.loc[self.stop_times_df['trip_id'] == trip_id]
            start_times.append(stop_times.loc[stop_times['stop_sequence'] == 1]['arrival_time'].to_numpy()[0])

        headways = []
        prev_time = start_times[0]
        for curr_time in start_times:
            print(prev_time, curr_time)
            headways.append(self.get_time_diff(prev_time, curr_time))

            prev_time = curr_time
        headways = headways[1:]
        avg_headway = np.mean(headways) / 60 # convert to minutes
        return avg_headway
    """

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    # Generate network, node file, network file                                                            #
    #                                                                               #
    # ----------------------------------------------------------------------------- #

    def make_network_df(self):
        '''
        Generates a network file dataframe

        Columns:
            from_node:  ID of the start node
            to_node:    ID of the end node
            distance:   Distance between nodes measured in miles
            travel_time:    Total time in seconds to travel between start and end nodes
            link_type:  Link type of the edge: 0 for transit link, -1 for walking link
            route_id:   ID or list of IDs of the route(s) that the start and end stops share
            route_type: Route type of the edge, defined by GTFS files, or -1 for walking link

        Returns:
            dataframe containing all network information
        
        '''

        self.network_df = pd.DataFrame(columns=['from_node', 'to_node','distance',
                                       'travel_time','link_type','route_id', 'route_type'])

        for route_times_df in self.route_times_dfs.values():
            for i, curr_row in route_times_df.iterrows():
                if i == len(route_times_df)-1:
                    continue
                next_row = route_times_df.iloc[i+1]
                from_node_id = int(curr_row['stop_id'])
                to_node_id = int(next_row['stop_id'])

                # If this edge has already been added, continue

                if from_node_id in self.network_df['from_node'].values:
                    if to_node_id in self.network_df['to_node'].values:
                        continue

                # Get the route_id(s) that both stops share
                from_routes = self.translate_id(from_node_id, 'stop', 'route')
                to_routes = self.translate_id(to_node_id, 'stop', 'route')
                route_ids = list(set(from_routes)&set(to_routes))

                
                route_id = route_ids[0]
                # remove the direction ID from the end of the route
                original_route_id = route_id[:-2]
                
                route_type = self.routes_df.loc[self.routes_df['route_id']==original_route_id]['route_type'].to_numpy()[0]
                dist = self.get_shape_dist(route_id, from_node_id, to_node_id)

                time = self.get_travel_time(route_id, from_node_id, to_node_id)

                # Add this pair to the network
                data = {'from_node':from_node_id,
                    'to_node':to_node_id,
                    'distance':dist,
                    'travel_time':time,
                    'link_type': 0,
                    'route_id':original_route_id,
                    'route_type':int(route_type)}

                self.network_df.loc[len(self.network_df)] = data

        # Add transfer links for all stops
        stop_ids = self.stops_df['stop_id'].to_numpy()
        for from_node_id in stop_ids:
            lat, lon = self.get_node_coords(from_node_id)
            nearby_stops = self.s_hash.get_nearby_stops(lat, lon)
            for to_node_id, dist in nearby_stops.items():
                to_node_id = int(float(to_node_id))

                route_id = -1 # walking links are not on a route
                
                route_type = -1 # walking link 
                
                time = (dist / self.walking_speed) * 3600 # time in seconds

                # Add stop to network_df
                data = {'from_node':from_node_id,
                    'to_node':to_node_id,
                    'distance':dist,
                    'travel_time':time,
                    'link_type': -1,
                    'route_id':route_id,
                    'route_type':int(route_type)}

                self.network_df.loc[len(self.network_df)] = data
        
        return self.network_df
    
    def make_node_file(self, txt_fn): 
        '''
        Creates csv node file from stops_df
        Node file contains information about all stops
        '''
        self.stops_df.to_csv(txt_fn, index=False, header=True)

    def make_network_file(self, txt_fn): 
        '''
        Creates csv network file from network_df
        Network file contains information about all edges in the network
        '''
        self.network_df.to_csv(txt_fn, index=False, header=True)

class S_Hash:
    '''
    Class that uses spatial hashing to speed up the process of finding nearby nodes
    '''
    def __init__(self,boundaries,thresh, stops_df):
        '''
        Sets latitude and longitude limits
        Creates number of buckets and step size based on thresh
        '''
        self.north = boundaries[0]
        self.south = boundaries[1]
        self.east = boundaries[2]
        self.west = boundaries[3]
        self.thresh = thresh
        
        height = geopy.distance.geodesic((self.north,self.west), (self.south,self.west)).mi
        width = geopy.distance.geodesic((self.north,self.west),(self.north,self.east)).mi

        
        self.num_buckets = int(math.ceil(max(height,width)) / thresh)

        # Determine step size for lat and lon
        self.step = max((abs(self.north - self.south) / self.num_buckets,
                    abs(self.west - self.east) / self.num_buckets))
        
        # Bucket index : (coordinates in that bucket, stop_id of coordinates)
        self.buckets = {}

        # Generate the S_Hash
        self.generate_hash(stops_df)
    
    def get_key(self, lat, lon):
        '''
        Gets the bucket key for a set of coordinates

        Args:
            lat (float): latitude of target location
            lon (float): longitude of the target location

        Returns:
            tuple: Key of the corresponding bucket
        '''
        lon_dist = abs(self.west - lon)
        lat_steps = math.floor(lon_dist/self.step)
        lat_dist = abs(self.north - lat)
        lon_steps = math.floor(lat_dist/self.step)
        return (lat_steps,lon_steps)
    
    def generate_hash(self, stops_df):
        '''
        Populates the hash with stops in stops_df

        Args:
            stops_df (dataframe): Dataframe containing all stop information
        '''
        # Add all nodes to the hash
        for i, row in stops_df.iterrows():
            stop_id, lat, lon = row['stop_id'], row['stop_lat'], row['stop_lon']
            key = self.get_key(lat,lon)
            if not key in self.buckets.keys():
                self.buckets[key] = np.vstack((['0',0,0],(stop_id, lat, lon)))
            else:
                self.buckets[key] = np.vstack((self.buckets[key],
                                               (stop_id, lat, lon)))
    
    def get_nearby_stops(self, lat, lon):
        '''
        Gets all stops within thresh distance of the input coordinates
        Distance measured in miles
        
        Args:
            lat (float): latitude of target location
            lon (float): longitude of the target location
        
        Returns:
            {stop_id: dist}: dict of nearby stop IDs and their distance (mi) to the target coordinates
        '''
        stops = {}
        coords1 = (lat,lon)
        key = self.get_key(lat,lon)
        if key not in self.buckets.keys():
            self.buckets[key] = np.array([])
        
        # Check coordinates in the same bucket
        for stop in self.buckets[key]:
            stop_id = stop[0]
            coords2 = (stop[1],stop[2])
            dist = geopy.distance.geodesic(coords1,coords2).mi
            if 0 < dist < self.thresh:
                stops[stop_id] = dist
        
        # Check coordinates in neighboring buckets (8 surrounding buckets)
        for i in range(-1,2,2):
            neigh_key = (key[0]+i,key[1])
            if neigh_key in self.buckets.keys():
                for stop in self.buckets[neigh_key]:
                    stop_id = stop[0]
                    coords2 = (stop[1],stop[2])
                    dist = geopy.distance.geodesic(coords1,coords2).mi
                    if 0 < dist < self.thresh:
                        stops[stop_id] = dist
            neigh_key = (key[0],key[1]+i)
            if neigh_key in self.buckets.keys():
                for stop in self.buckets[neigh_key]:
                    stop_id = stop[0]
                    coords2 = (stop[1],stop[2])
                    dist = geopy.distance.geodesic(coords1,coords2).mi
                    if 0 < dist < self.thresh:
                        stops[stop_id] = dist
            neigh_key = (key[0]+i,key[1]+i)
            if neigh_key in self.buckets.keys():
                for stop in self.buckets[neigh_key]:
                    stop_id = stop[0]
                    coords2 = (stop[1],stop[2])
                    dist = geopy.distance.geodesic(coords1,coords2).mi
                    if 0 < dist < self.thresh:
                        stops[stop_id] = dist
        neigh_key = (key[0]+1,key[1]-1)
        if neigh_key in self.buckets.keys():
            for stop in self.buckets[neigh_key]:
                stop_id = stop[0]
                coords2 = (stop[1],stop[2])
                dist = geopy.distance.geodesic(coords1,coords2).mi
                if 0 < dist < self.thresh:
                    stops[stop_id] = dist
        neigh_key = (key[0]-1,key[1]+1)
        if neigh_key in self.buckets.keys():
            for stop in self.buckets[neigh_key]:
                stop_id = stop[0]
                coords2 = (stop[1],stop[2])
                dist = geopy.distance.geodesic(coords1,coords2).mi
                if 0 < dist < self.thresh:
                    stops[stop_id] = dist
        return stops