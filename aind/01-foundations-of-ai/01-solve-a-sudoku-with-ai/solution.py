assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        values(dict): the values dictionary with the naked twins eliminated from peers.
    """
    # Find all instances of naked twins
    naked_twin_dict = {}
    for unit in unitlist:
        # Build a dictionary or hash map to identify a naked twin pair
        vdict = {}
        for box in unit:
            # Identify box containing only 2 possibilities
            # as a candidate for a naked twin
            if len(values[box]) == 2:
                if not values[box] in vdict:
                    vdict[values[box]] = [box]
                else:
                    vdict[values[box]].append(box)
        # Examine the dictionary to validate the candidates present as
        # naked twin pairs
        for key in vdict:
            # Condition for the candidate to be a naked twin pair
            if len(vdict[key]) == 2:
                if not key in naked_twin_dict:
                    naked_twin_dict[key] = [unit]
                else:
                    naked_twin_dict[key].append(unit)

    # Eliminate the naked twins as possibilities for their peers
    for key in naked_twin_dict:
        for unit in naked_twin_dict[key]:
            for box in unit:
                if values[box] != key:
                    assign_value(values, box, values[box].replace(key[0], ''))
                    assign_value(values, box, values[box].replace(key[1], ''))
    return values

def naked_triplet(values):
    """Eliminate values using the naked triplet strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        values(dict): the values dictionary with the naked triplets eliminated from peers.
    """
    # Find all instances of naked triplets
    naked_triplet_dict = {}
    for unit in unitlist:
        # Build a dictionary or hash map to identify a naked triplet
        vdict = {}
        for box in unit:
            # Identify box containing only 3 possibilities
            # as a candidate for a naked triplet
            if len(values[box]) == 3:
                if not values[box] in vdict:
                    vdict[values[box]] = [box]
                else:
                    vdict[values[box]].append(box)
        # Examine the dictionary to validate the candidates present as
        # naked triplets
        for key in vdict:
            # Condition for the candidate to be a naked triplet
            if len(vdict[key]) == 3:
                if not key in naked_triplet_dict:
                    naked_triplet_dict[key] = [unit]
                else:
                    naked_triplet_dict[key].append(unit)

    # Eliminate the naked triplet as possibilities for their peers
    for key in naked_triplet_dict:
        for unit in naked_triplet_dict[key]:
            for box in unit:
                if values[box] != key:
                    assign_value(values, box, values[box].replace(key[0], ''))
                    assign_value(values, box, values[box].replace(key[1], ''))
                    assign_value(values, box, values[box].replace(key[2], ''))
    return values


# Development ongoing!
''' TODO: differentiate between triplets and twins
    keep track and compare after
    - create two distinct dictionaries for twins and triplets
    - compare the twins and triplets to the triplets in the library for
    each unit
    - track the amount of matching boxes (e.g. [3,4,8], [3,4], [3,8])
    - use the identified entry in the triplet library to eliminate
    digits in other boxes (3,4,8)
    STATUS: not completed
'''
def naked_twinlet(values):
    """Eliminate values using the combination of naked twin and naked
    triplet strategy ('twinlet strategy').
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        values(dict): the values dictionary with the 'naked twinlets' eliminated from peers.
    """
    # Find all instances of 'naked twinlets'
    naked_twinlet_dict = {}
    for unit in unitlist:
        # Build two dictionaries or hash maps to identify twin or triplet
        # contributors to the 'naked twinlet'
        vdict_triplets = {}
        vdict_twins = {}
        for box in unit:
            # Identify box containing only 2 or 3 possibilities
            # as a candidate for a 'naked twinlet'
            if len(values[box]) == 3:
                if not values[box] in vdict_triplets:
                    vdict_triplets[values[box]] = [box]
                else:
                    vdict_triplets[values[box]].append(box)
            if len(values[box]) == 2:
                if not values[box] in vdict_twins:
                    vdict_twins[values[box]] = [box]
                else:
                    vdict_twins[values[box]].append(box)
        # Examine the dictionaries to validate the candidates present as
        # 'naked twinlets'
        for key in vdict_triplets:
            # Condition for the candidate to be a naked triplet
            '''
            TODO: Compare keys in both dictionaries to keys in vdict_triplets
            in order to identify 3 boxes in a unit representing a
            'naked twinlet'
            Strategy: compare every library key for every box in a unit
            to other keys and increment a temp_counter. If the temp_counter
            is 2 for twins or 3 for triplets we increase the twinlet_counter
            for the key. If the twinlet_counter reaches 2, we've identified
            a 'naked twinlet' and can add it to naked_twinlet_dict. 
            '''           
    return values

def cross(A, B):
    """Cross product of elements in a and elements in b."""
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

# Define 2 more (diagonal) units and add them to 'unitlist' in order to solve diagonal Sudokus:
# diagonal_unit_topdown = [['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8', 'I9']]
# diagonal_unit_bottomup = [['A9', 'B8', 'C7', 'D6', 'E5', 'F4', 'G3', 'H2', 'I1']]
diagonal_units = [[x+y for x, y in zip(rows, cols)], [x+y for x, y in zip(rows, cols[::-1])]]
unitlist = row_units + column_units + square_units + diagonal_units

# Old unitlist:
# unitlist = row_units + column_units + square_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    all_digits = '123456789'
    for keys in grid:
        if keys == '.':
            values.append(all_digits)
        elif keys in all_digits:
            values.append(keys)
    assert len(values) == 81
    return dict(zip(boxes, values))

def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    """Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values

def reduce_puzzle(values):
    """
    Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same, return the sudoku.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, try all possible values."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and 
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    values = search(values)
    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
