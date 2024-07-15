For all the tasks present in the dataset, there is no clear categorization based on a set of base attributes which indicate the change/transformation the input goes through to get the output.
The base set of attributes that can be considered are:
*   number of squares -> how they change? same, doubles, triples, halve, etc.
*   colors -> Does the colors change or not? mono-color or multi-color, etc. Here, black is equivlent to no-color.
*   transformations -> how pattern changes -> rotation, mirror, multiply, etc.

The tasks selected make the dataset considerably simpler in complexity compared to the original ARC-AGI dataset. We list the subset of tasks by their task-Id and describe their transformation:

1. Fill the holes with a single color.
2. Change the color - mono-color
5. or operation and change color
6. complete the pattern - fill the board with repeating patterns
9. mono-color stripes to multi-color - grey to rbgy
12. complete with stripes - fill the strip horizontal or verticle with the corresponding color
15. stripes - color map
20. Simpliplification - each segment is reduced to one square.
27. Pickup 2 colors and generate fixed shape using both colors.
30. Trim the squares to minimum
52. Move shape in particular direction.
84. Punch holes at the center.
85. Add four flaps to a square
94. Surround a square
97. Hollow out the square/rectangle at the center.
99. Select the bigger area.
128. Fill squares with max number of colors appearing in 3x3
152. Attach the shapes - needs spactial awareness, orientation and matching - change in canvas
159. Identify shape and change color if it is a +
177. From a square/rectangle of color strips take the single strip cross-section capturing all the colors - canvas size change
372. 2xN matrix with 2 color horizontal stripes. Transforms to alternating colors in adjacent tiles.

List of tasks which are not present in the ARC-AGI dataset but similar in style and flavor:

1. Complement the squares with the same color
2. Complement the squares with a different color
3. Fill all the squares with the same color
4. Fill squares with the max number of colors appearing in 3x3
5. Nx2 matrix with 2 color vertical stripes. Transforms to alternating colors in adjacent tiles
6. Convert an X to an O
7. Convert an X to an O and change color
