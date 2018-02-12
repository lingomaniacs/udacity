# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: *Constraint propagation is used in the 'Naked Twins Problem' as follows:*

1. Every unit containing naked twins has to be identified. This means we're looking for 2 boxes having identical sets of only 2 possible candidates in the same unit.

2. Because of the fact that the 2 values within the naked twin pair are limited to those 2 boxes, none of the other boxes in its unit can have either one of them as candidates. 

3. Hence those values can be eliminated from the peers of the naked twin pair in every related unit.

4. This can lead to either newly identified single choices or other naked twin pairs after such an elimination.

5. By iteration of the eliminate(), naked_twins() and only_choice() strategies the constraint propagation is formed as part of our sudoku solving algorithm. 

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: *Constraint propagation is used to solve a 'Diagonal Sudoku Problem' as follows:*

1. Boxes belonging to the two diagonal units need to be idtentified.

2. We consider the diagonal units in every strategy we are using (eliminate(), naked_twins() and only_choice()) along with the other units (row, column, square) as part of the constraint propagation for solving a diagonal sudoku problem.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

