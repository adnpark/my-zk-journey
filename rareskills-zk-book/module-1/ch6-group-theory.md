# Elementary Group Theory for Programmers

## Intro

A group is a set with:

-   a closed binary operator
-   the binary operator is also associative
-   an identity element
-   every element having an inverse

## Examples of groups

# Exercise

## As an exercise for the reader, show that the set {1} with the binary 'x' operator is a group.

-   closed binary operator: 1 x 1 = 1
-   associative: 1 x (1 x 1) = (1 x 1) x 1
-   identity element: 1 x 1 = 1
-   inverse: 1 x 1 = 1

## Exercise: Integers (positive and negative) are not a group under multiplication. Show which of the four requirements (closed, associative, existence of identity, all elements having an inverse) are not satisfied.

-   closed: you can get an integer when you multiply two integers
-   associative: (a \* b) \* c = a \* (b \* c)
-   identity: 1 \* a = a \* 1 = a
-   inverse: 2 \* x = 1, x = 1/2, which is not an integer
