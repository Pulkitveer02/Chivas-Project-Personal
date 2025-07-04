# Mapping dictionaries for typeId and qualifierId

qualifier_dict = {
    1: 'Long ball',
    2: 'Cross',
    3: 'Head pass',
    4: 'Through ball',
    5: 'Free kick taken',
    6: 'Corner taken',
    7: 'Players caught offside',
    8: 'Goal disallowed',
    9: 'Penalty',
    10: 'Handball',
    11: '6-seconds violation',
    12: 'Dangerous play',
    13: 'Foul',
    14: 'Last line',
    15: 'Head',
    16: 'Small box - Centre',
    17: 'Box - Centre',
    18: 'Out of box - Centre',
    19: '35+ Centre',
    20: 'Right footed',
    21: 'Other body part',
    22: 'Regular play',
    23: 'Fast break',
    24: 'Set piece',
    25: 'From corner',
    26: 'Free kick',
    28: 'Own Goal',
    29: 'Assisted',
    30: 'Involved',
    31: 'Yellow Card',
    32: 'Second yellow',
    33: 'Red Card',
    34: 'Referee abuse',
    35: 'Argument',
    36: 'Violent Conduct',
    37: 'Time wasting',
    38: 'Excessive celebration',
    39: 'Crowd interaction',
    40: 'Other reason',
    41: 'Injury',
    42: 'Tactical',
    43: 'Deleted event',
    44: 'Player position',
    45: 'Temperature',
    46: 'Conditions',
    47: 'Field Pitch',
    48: 'Lightings',
    49: 'Attendance figure',
    50: 'Official position',
    51: 'Official ID',
    52: 'Possession time',
    53: 'Injured player ID',
    54: 'End cause',
    55: 'Related event ID',
    56: 'Zone',
    57: 'End type',
    58: 'Temp stop status',
    59: 'Jersey Number',
    60: 'Small box - Right',
    61: 'Small box - Left',
    62: 'Box - Deep Right',
    63: 'Box - Right',
    64: 'Box - Left',
    65: 'Box - Deep Left',
    66: 'Out of box - Deep Right',
    67: 'Out of box - Right',
    68: 'Out of box - Left',
    69: 'Out of box - Deep Left',
    70: '35+ Right',
    71: '35+ Left',
    72: 'Left footed',
    73: 'Left',
    74: 'High',
    75: 'Right',
    76: 'Low Left',
    77: 'High Left',
    78: 'Low Centre',
    79: 'High Centre',
    80: 'Low Right',
    81: 'High Right',
    82: 'Blocked',
    83: 'Close Left',
    84: 'Close Right',
    85: 'Close High',
    86: 'Close Left and High',
    87: 'Close Right and High',
    88: 'High claim',
    89: '1 on 1',
    90: 'Deflected save',
    91: 'Dive and deflect',
    92: 'Catch',
    93: 'Dive and catch',
    94: 'Def block',
    95: 'Back pass',
    96: 'Corner situation',
    97: 'Direct free',
    98: 'Pitch X Coordinate',
    99: 'Pitch Y Coordinate',
    100: 'Six Yard Blocked',
    101: 'Saved Off Line',
    102: 'Goalmouth Y Coordinate',
    103: 'Goalmouth Z Coordinate',
    104: 'Attempt Position X Coordinate',
    105: 'Attempt Position Y Coordinate',
    106: 'Attacking Pass',
    107: 'Throw in',
    108: 'Volley',
    109: 'Overhead',
    110: 'Half Volley',
    111: 'Diving Header',
    112: 'Scramble',
    113: 'Strong',
    114: 'Weak',
    115: 'Rising',
    116: 'Dipping',
    117: 'Lob',
    118: 'One Bounce',
    119: 'Few Bounces',
    120: 'Swerve Left',
    121: 'Swerve Right',
    122: 'Swerve Moving',
    123: 'Keeper Throw',
    124: 'Goal Kick',
    125: 'Free Kick Position X Coordinate',
    126: 'Free Kick Position Y Coordinate',
    127: 'Direction of Play',
    128: 'Punch',
    129: 'Ten Minute Possession',
    130: 'Team Formation',
    131: 'Team Player Formation',
    132: 'Simulation',
    133: 'Deflection',
    134: 'Far Wide Left',
    135: 'Far Wide Right',
    136: 'Keeper Touched',
    137: 'Keeper Saved',
    138: 'Hit Woodwork',
    139: 'Own Player',
    140: 'Pass End X',
    141: 'Pass End Y',
    142: 'Flag to Checker',
    143: 'Star Rating',
    144: 'Deleted Event Type',
    145: 'Formation slot',
    146: 'Blocked X Coordinate',
    147: 'Blocked Y Coordinate',
    148: 'Danger',
    149: 'Inside',
    150: 'Outside',
    151: 'Short',
    152: 'Direct',
    153: 'Not past goal line',
    154: 'Intentional Assist',
    155: 'Chipped',
    156: 'Lay-off',
    157: 'Launch',
    158: 'Persistent Infringement',
    159: 'Foul and Abusive Language',
    160: 'Throw-in set piece',
    161: 'Encroachment',
    162: 'Leaving field',
    163: 'Entering field',
    164: 'Spitting',
    165: 'Professional Foul Last Man',
    166: 'Professional Foul Handball',
    167: 'Out of play',
    168: 'Flick-on',
    169: 'Leading to attempt',
    170: 'Leading to goal',
    171: 'Rescinded Card',
    173: 'Parried safe',
    174: 'Parried danger',
    175: 'Fingertip',
    176: 'Caught',
    177: 'Collected',
    178: 'Standing',
    179: 'Diving',
    180: 'Stooping',
    181: 'Reaching',
    182: 'Hands',
    183: 'Feet',
    184: 'Dissent',
    185: 'Blocked cross',
    186: 'Scored',
    187: 'Saved',
    188: 'Missed',
    189: 'Not visible',
    190: 'From shot off target',
    191: 'Off the ball foul',
    192: 'Block by hand',
    193: 'Goal measure',
    194: 'Captain',
    195: 'Pull back',
    196: 'Switch of play',
    197: 'Team kit',
    198: 'GK hoof',
    199: 'GK kick from hands',
    200: 'Referee stop',
    201: 'Referee delay',
    202: 'Weather problem',
    203: 'Crowd trouble',
    204: 'Fire',
    205: 'Object thrown on pitch',
    206: 'Spectator on pitch',
    207: "Awaiting official's decision",
    208: 'Referee injury',
    209: 'Game end',
    210: 'Assist',
    211: 'Overrun',
    212: 'Length',
    213: 'Angle',
    214: 'Big chance',
    215: 'Individual play',
    216: '2nd related event ID',
    217: '2nd assisted',
    218: '2nd assist',
    219: 'Players on both posts',
    220: 'Player on near post',
    221: 'Player on far post',
    222: 'No players on posts',
    223: 'In-swinger',
    224: 'Out-swinger',
    225: 'Straight',
    226: 'Suspended',
    227: 'Resume',
    228: 'Own shot blocked',
    229: 'Post match complete',
    230: 'GK X Coordinate',
    231: 'GK Y Coordinate',
    232: 'Unchallenged',
    233: 'Opposite related event ID',
    234: 'Home Team Possession',
    235: 'Away Team Possession',
    236: 'Blocked pass',
    237: 'Low',
    238: 'Fair Play',
    239: 'By Wall',
    240: 'GK Start',
    241: 'Indirect',
    242: 'Obstruction',
    243: 'Unsporting behaviour',
    244: 'Not Retreating',
    245: 'Serious Foul',
    246: 'Drinks Break',
    247: 'Offside',
    248: 'Goal line',
    249: 'Temp Shot On',
    250: 'Temp Blocked',
    251: 'Temp Post',
    252: 'Temp Missed',
    253: 'Temp Miss Not Passed Goal Line',
    254: 'Follows a Dribble',
    255: 'Open Roof',
    256: 'Air Humidity',
    257: 'Air Pressure',
    258: 'Sold Out',
    259: 'Celsius degrees',
    260: 'Floodlight',
    261: '1 on 1 chip',
    262: 'Back heel',
    263: 'Direct corner',
    264: 'Aerial Foul',
    265: 'Attempted Tackle',
    266: 'Put Through',
    267: 'Right Arm',
    268: 'Left Arm',
    269: 'Both Arms',
    270: 'Right Leg',
    271: 'Both Legs',
    273: 'Hit Right Post',
    274: 'Hit Left Post',
    275: 'Hit Bar',
    276: 'Out on sideline',
    277: 'Minutes',
    278: 'Tap',
    279: 'Kick Off',
    280: 'Fantasy Assist Type',
    281: 'Fantasy Assisted By',
    282: 'Fantasy Assist Team',
    283: 'Coach ID',
    284: 'Duel',
    285: 'Defensive',
    286: 'Offensive',
    287: 'Over-arm',
    288: 'Out of Play Secs',
    289: 'Denied goal-scoring opp',
    290: 'Coach types',
    291: 'Other Ball Contact Type',
    292: 'Detailed Position ID',
    293: 'Position Side ID',
    294: 'Shove/Push',
    295: 'Shirt Pull/Holding',
    297: 'Follows Shot Rebound',
    298: 'Follows Shot Blocked',
    299: 'Clock Affecting',
    300: 'Solo Run',
    301: 'Shot from cross',
    302: 'Checks complete/Live collection checks complete',
    303: 'Floodlight failure',
    304: 'Ball In Play',
    305: 'Ball Out of Play',
    306: 'Kit change',
    307: 'Phase of possession ID',
    308: 'Goes to Extra Time',
    309: 'Goes to Penalties',
    310: 'Player goes out',
    311: 'Player comes back',
    312: 'Phase of possession start',
    313: 'Illegal Restart',
    314: 'End of Offside',
    316: 'Passed Penalty',
    317: 'Penalty Set Piece',
    319: 'Captain change',
    323: 'Follows a Rebound',
    324: 'Follows a Take On',
    325: 'Abandonment To Follow',
    328: 'First Touch',
    329: 'VAR - Goal Awarded',
    330: 'VAR - Penalty Awarded',
    331: 'VAR - Penalty Not Awarded',
    332: 'VAR - (Red) Card Upgrade',
    333: 'VAR - Mistaken Identity',
    334: 'VAR - Other',
    335: 'Referee Decision Confirmed',
    336: 'Referee Decision Cancelled',
    338: 'Follows a Rebound Event ID',
    341: 'VAR - Goal Not Awarded',
    342: 'VAR - Red Card Given',
    343: 'Review',
    344: 'Video coverage lost',
    345: 'Overhit cross',
    346: 'Next event Goal-Kick',
    347: 'Next event Throw-In',
    348: 'Penalty taker ID',
    349: 'Goalkeeper punch outcome',
    353: 'Second (2nd) opposite related event ID',
    354: 'Ball hits referee',
    355: 'Entering referee review area',
    356: 'Excessive usage of review signal',
    357: 'Entering video operations room',
    358: 'Official body: Reviewed and confirmed',
    359: 'Official body: Reviewed and changed',
    361: 'Incorrect out of play decision',
    362: 'Viral',
    363: 'Away attendance',
    364: 'VAR Delay',
    365: 'Reviewed event ID',
    374: 'Goal shot timestamp',
    375: 'Goal shot game clock',
    376: 'Low GK intervention',
    377: 'Medium GK intervention',
    378: 'High GK intervention',
    380: 'Other obstacle',
    381: 'Fumble',
    383: 'Touch type control',
    384: 'Touch type pass',
    385: 'Touch type clearance',
    386: 'Driven cross',
    387: 'Floated cross',
    388: 'Jumping',
    389: 'Sliding',
    390: 'Causing player',
    391: 'Mis-hit',
    392: 'Reckless offence',
    393: 'Tactical Foul',
    394: 'Corner not taken',
    395: 'GK x coordinate time of goal',
    396: 'GK y coordinate time of goal',
    397: 'Blocked clearance',
    398: 'GK Challenge',
    399: 'Intended tackle target',
    406: 'Collection complete',
    436: 'Pre-Review Event Type',
    458: 'Not assisted',
    459: 'Event type review',
    464: 'Take on space',
    465: 'Take on overtake',
    467: 'Defensive 1 v 1',
    468: 'Related error 1 ID',
    472: 'Fantasy assist ID\xa0',
    474: 'Related error 2 ID',
    476: 'New start time',
    478: 'Officially announced',
    479: 'Estimated',
    484: 'Dubious scorer',
    485: 'Advantage played',
}

typeid_dict = {
    1: 'Pass',
    2: 'Offside Pass',
    3: 'Take On',
    4: 'Foul',
    5: 'Out',
    6: 'Corner Awarded',
    7: 'Tackle',
    8: 'Interception',
    10: 'Save',
    11: 'Claim',
    12: 'Clearance',
    13: 'Miss',
    14: 'Post',
    15: 'Attempt Saved',
    16: 'Goal',
    17: 'Card',
    18: 'Player off',
    19: 'Player on',
    20: 'Player retired',
    21: 'Player returns',
    22: 'Player becomes goalkeeper',
    23: 'Goalkeeper becomes player',
    24: 'Condition change',
    25: 'Official change',
    27: 'Start delay',
    28: 'End delay',
    29: 'Temporary stop',
    30: 'End',
    31: 'Picked an orange',
    32: 'Start',
    34: 'Team set up',
    36: 'Player changed Jersey number',
    37: 'Collection End',
    38: 'Temp Goal',
    39: 'Temp Attempt',
    40: 'Formation change',
    41: 'Punch',
    42: 'Good skill',
    43: 'Deleted event',
    44: 'Aerial',
    45: 'Challenge',
    46: 'Postponed',
    47: 'Rescinded card',
    49: 'Ball recovery',
    50: 'Dispossessed',
    51: 'Error',
    52: 'Keeper pick-up',
    53: 'Cross not claimed',
    54: 'Smother',
    55: 'Offside provoked',
    56: 'Shield ball opp',
    57: 'Foul throw-in',
    58: 'Penalty faced',
    59: 'Keeper Sweeper',
    60: 'Chance missed',
    61: 'Ball touch',
    62: 'Event placeholder',
    63: 'Temp Save',
    64: 'Resume',
    65: 'Contentious referee decision',
    66: 'Possession Data',
    67: '50/50',
    68: 'Referee Drop Ball',
    69: 'Failed to Block',
    70: 'Injury Time Announcement',
    71: 'Coach Setup',
    72: 'Caught Offside',
    73: 'Other Ball Contact',
    74: 'Blocked Pass',
    75: 'Delayed start',
    76: 'Early end',
    78: 'Temp card',
    79: 'Coverage interruption',
    80: 'Drop of Ball',
    81: 'Obstacle',
    82: 'Control',
    83: 'Attempted tackle',
    84: 'Deleted After Review',
}
