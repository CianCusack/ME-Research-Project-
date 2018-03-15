import math

# Calculate slope of the line between two given points
def slope(p1, p2, accuracy):

    m = float(0)

    # Ensure denominator does not go to zero
    if p2[0]-p1[0] != 0:

        # Slope formula
        m = float((p2[1]-p1[1])/(p2[0]-p1[0]))

        # Round to two decimal places as too much accuracy reduces chances of line crossing
        m = round(m, accuracy)

    return m

