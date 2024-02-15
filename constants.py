old_size_dict = {
    0.25 : 0.25,
    0.50: 0.50,
    0.75 : 0.75,
    1.00 : 1.0,
    2.00 : 2.0,
    3.00 : 3.0,
    4.00 : 3.5,
    5.00 : 4.0,
    6.00 : 4.5,
    7.00 : 5.0,
    8.00 : 6.0,
    9.00 : 7.0,
    10.00 : 7.5,
    11.00 : 8.0,
    12.00 : 9.0,
    13.00 : 9.5,
    21.00 : 15.0,
    33.00 : 23.0,
    48.00 : 34.0
}
#the new size dict has spots that are smaller. I changed the window size.
new_size_dict = {
    1.0 : 0.75,
    2.0 : 1.00,
    3.0 : 1.25,
    4.0 : 1.50,
    5.0 : 2.00,
    6.0 : 2.25,


}


StimDict = {
    "1" : "1",
    "2" : "2",
    "3" : "3",
    "7" : "7",
    "9" : "9",
    "12" : "12",

}

monocular_dict = {
    0: [1, 0.25, 0],
    1: [0, 0.25, 1],
    2: [0, 1, 0],
    3: [1, 0, 1],
    4: [0, 0.75, 1],
    5: [0.75, 1, 0],
    6: [0.25, 0, 1],
    7: [1, 0, 0.25],
    8: [0.25, 1, 0],
    9: [0, 0.75, 1],
    10: [1, 0, 0.75],
}

nulldict = {
    "right": "left",
    "left": "right",
    "forward": "backward",
    "backward": "forward",
    "forward_right": "backward_left",
    "backward_left": "forward_right",
    "forward_left": "backward_right",
    "backward_right": "forward_left",
}

baseBinocs = [
    "medial_left",
    "lateral_left",
    "medial_right",
    "lateral_right",
    "diverging",
    "converging",
]

deg_dict = {
    "right": 90,
    "forward_right": 45,
    "forward": 0,
    "forward_left": 315,
    "left": 270,
    "backward_left": 215,
    "backward": 180,
    "backward_right": 135,
}

#%%
