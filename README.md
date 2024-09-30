# Transit-Supernetwork
ProcessGTFS.py includes two classes: GTFSNetwork and S_Hash. These classes enable the creation of node and network csv files which can be used to generate traversable graphical networks from GTFS files.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Clone the repository
2. Install dependencies:
    pip install -r requirements.txt

## Usage

An example notebook ```Processing GTFS Example.ipynb``` is included.

Processing requries the following GTFS files:
1. stop_times.txt
2. stops.txt
3. trips.txt
4. routes.txt
5. shapes.txt

### Class GTFSNetwork

This class stores all necessary data from the provided GTFS files and generates the speeds and travel times for all edges in the network.

### Class S_Hash

This class defines spatial hashes which are used to quickly find stops within a defined threshold. The desired threshold must be defined at initialization. 
