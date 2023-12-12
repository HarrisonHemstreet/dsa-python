## Search
### Linear Search:
Best: O(1)
Average: O(n)
Worst: O(n)
##### Steps:
1. This function accepts an array and a value
2. Loop through the array and check if the current array element is equal to the value
3. If it is, return the index at which the element is found
4. If the value is never found, return -1
### Binary Search:
Best: O(1)
Average: O(log n)
Worst: O(log n)
##### Steps:
1. This function accepts a sorted array and a value
2. Create a left pointer at the start of the array, and a right pointer at the end of the array
3. While the left pointer comes before the right pointer:
4. Create a pointer in the middle
5. If you find the value you want, return the index
6. If the value is too small, move the left pointer up
7. If the value is too large, move the right pointer down
8. If you never find the value, return -1
### Naive String Search:
Best: O(nm)
##### Steps:
1. Loop over the longer string
2. Loop over the shorter string
3. If the characters don't match, break out of the inner loop
4. If the characters do match, keep going
5. If you complete the inner loop and find a match, increment the count of matches
6. Return the count
### KMP String Search:
Best: O(n + m) time, O(m) space
##### Description:
The Knutt-Morris-Pratt algorithm offers an improvement over the naive approach
Published in 1977
This algorithm more intelligently traverses the longer string to reduce the amount of redundant searching
## Sort
### Bubble Sort:
Best: O(n)
Average: O(n^2)
Worst: O(n^2)
Space: O(1)
##### Steps:
1. Start looping from with a variable called i the end of the array towards the beginning
2. Start an inner loop with a variable called j from the beginning until i - 1
3. If arr[j] is greater than arr[j+1], swap those two values!
4. Return the sorted array
### Selection Sort:
Best: O(n^2)
Average: O(n^2)
Worst: O(n^2)
Space: O(1)
##### Steps:
1. Store the first element as the smallest value you've seen so far.
2. Compare this item to the next item in the array until you find a smaller number.
3. If a smaller number is found, designate that smaller number to be the new "minimum" and continue until the end of the array.
4. If the "minimum" is not the value (index) you initially began with, swap the two values.
5. Repeat this with the next element until the array is sorted.
### Insertion Sort:
Best: O(n)
Average: O(n^2)
Worst: O(n^2)
Space: O(1)
##### Steps:
1. Start by picking the second element in the array
2. Now compare the second element with the one before it and swap if necessary.
3. Continue to the next element and if it is in the incorrect order, iterate through the sorted portion (i.e. the left side) to place the element in the correct place.
4. Repeat until the array is sorted.
### Merge Sort:
Best: O(n log n)
Average: O(n log n)
Worst: O(n log n)
Space: O(n)
##### Steps:
1. Break up the array into halves until you have arrays that are empty or have one element
2. Once you have smaller sorted arrays, merge those arrays with other sorted arrays until you are back at the full length of the array
3. Once the array has been merged back together, return the merged (and sorted!) array
##### Merge Arrays:
1. Create an empty array, take a look at the smallest values in each input array
2. While there are still values we haven't looked at...
3. If the value in the first array is smaller than the value in the second array, push the value in the first array into our results and move on to the next value in the first array
4. If the value in the first array is larger than the value in the second array, push the value in the second array into our results and move on to the next value in the second array
5. Once we exhaust one array, push in all remaining values from the other array
### Quick Sort:
Best: O(n log n)
Average: O(n log n)
Worst: O(n^2)
Space: O(log n)
##### Steps:
1. Call the pivot helper on the array
2. When the helper returns to you the updated pivot index, recursively call the pivot helper on the subarray to the left of that index, and the subarray to the right of that index
3. Your base case occurs when you consider a subarray with less than 2 elements
##### Pivot Steps:
1. It will help to accept three arguments: an array, a start index, and an end index (these can default to 0 and the array length minus 1, respectively)
2. Grab the pivot from the start of the array 
3. Store the current pivot index in a variable (this will keep track of where the pivot should end up)
4. Loop through the array from the start until the end
5. If the pivot is greater than the current element, increment the pivot index variable and then swap the current element with the element at the pivot index
6. Swap the starting element (i.e. the pivot) with the pivot index
7. Return the pivot index
### Radix Sort:
Best: O(nk)
Average: O(nk)
Worst: O(nk)
Space: O(n + k)
n - length of array
k - number of digits(average)
##### Steps:
1. Define a function that accepts list of numbers
2. Figure out how many digits the largest number has
3. Loop from k = 0 up to this largest number of digits
4. For each iteration of the loop:
5. Create buckets for each digit (0 to 9)
6. place each number in the corresponding bucket based on its kth digit
7. Replace our existing array with values in our buckets, starting with 0 and going up to 9
8. return list at the end!
## Data Structures:
### Singly Linked List:
#### Methods:
##### Push:
1. This function should accept a value
2. Create a new node using the value passed to the function
3. If there is no head property on the list, set the head and tail to be the newly created node
4. Otherwise set the next property on the tail to be the new node and set the tail property on the list to be the newly created node
5. Increment the length by one
6. Return the linked list. 
##### Pop:
1. If there are no nodes in the list, return undefined
2. Loop through the list until you reach the tail
3. Set the next property of the 2nd to last node to be null
4. Set the tail to be the 2nd to last node
5. Decrement the length of the list by 1
6. Return the value of the node removed
##### Shift:
1. If there are no nodes, return undefined
2. Store the current head property in a variable
3. Set the head property to be the current head's next property
4. Decrement the length by 1
5. Return the value of the node removed
##### Unshift:
1. This function should accept a value
2. Create a new node using the value passed to the function
3. If there is no head property on the list, set the head and tail to be the newly created node
4. Otherwise set the newly created node's next property to be the current head property on the list
5. Set the head property on the list to be that newly created node
6. Increment the length of the list by 1
7. Return the linked list
##### Get:
This function should accept an index
If the index is less than zero or greater than or equal to the length of the list, return null
Loop through the list until you reach the index and return the node at that specific index
##### Set:
1. This function should accept a value and an index
2. Use your get function to find the specific node.
3. If the node is not found, return false
4. If the node is found, set the value of that node to be the value passed to the function and return true
##### Insert:
1. If the index is less than zero or greater than the length, return false
2. If the index is the same as the length, push a new node to the end of the list
3. If the index is 0, unshift a new node to the start of the list
4. Otherwise, using the get method, access the node at the index - 1
5. Set the next property on that node to be the new node
6. Set the next property on the new node to be the previous next
7. Increment the length
8. Return true
##### Remove:
1. If the index is less than zero or greater than the length, return undefined
2. If the index is the same as the length-1, pop
3. If the index is 0, shift
4. Otherwise, using the get method, access the node at the index - 1
5. Set the next property on that node to be the next of the next node
6. Decrement the length
7. Return the value of the node removed
##### Reverse:
1. Swap the head and tail
2. Create a variable called next
3. Create a variable called prev
4. Create a variable called node and initialize it to the head property
5. Loop through the list
6. Set next to be the next property on whatever node is
7. Set the next property on the node to be whatever prev is
8. Set prev to be the value of the node variable
9. Set the node variable to be the value of the next variable
10. Once you have finished looping, return the list
### Doubly Linked List:
#### Methods:
##### Push:
1. Create a new node with the value passed to the function
2. If the head property is null set the head and tail to be the newly created node 
3. If not, set the next property on the tail to be that node
4. Set the previous property on the newly created node to be the tail
5. Set the tail to be the newly created node
6. Increment the length
7. Return the Doubly Linked List
##### Pop:
1. If there is no head, return undefined
2. Store the current tail in a variable to return later
3. If the length is 1, set the head and tail to be null
4. Update the tail to be the previous Node.
5. Set the newTail's next to null
6. Decrement the length
7. Return the value removed
##### Shift:
1. If length is 0, return undefined
2. Store the current head property in a variable (we'll call it old head)
3. If the length is one
	* set the head to be null
	* set the tail to be null
4. Update the head to be the next of the old head
5. Set the head's prev property to null
6. Set the old head's next to null
7. Decrement the length
8. Return old head
##### Unshift:
1. Create a new node with the value passed to the function
2. If the length is 0:
	* Set the head to be the new node
	* Set the tail to be the new node
3. Otherwise:
	* Set the prev property on the head of the list to be the new node
	* Set the next property on the new node to be the head property 
	* Update the head to be the new node
4. Increment the length
5. Return the list
##### Get:
1. If the index is less than 0 or greater or equal to the length, return null
2. If the index is less than or equal to half the length of the list
	* Loop through the list starting from the head and loop towards the middle
	* Return the node once it is found
3. If the index is greater than half the length of the list
	* Loop through the list starting from the tail and loop towards the middle
	* Return the node once it is found
##### Set:
1. Create a variable which is the result of the get method at the index passed to the function
	* If the get method returns a valid node, set the value of that node to be the value passed to the function
	* Return true
2. Otherwise, return false
##### Insert:
1. If the index is less than zero or greater than or equal to the length return false
2. If the index is 0, unshift
3. If the index is the same as the length, push
4. Use the get method to access the index -1
5. Set the next and prev properties on the correct nodes to link everything together
6. Increment the length 
7. Return true
##### Remove:
1. If the index is less than zero or greater than or equal to the length return undefined
2. If the index is 0, shift
3. If the index is the same as the length-1, pop
4. Use the get method to retrieve the item to be removed
5. Update the next and prev properties to remove the found node from the list
6. Set next and prev to null on the found node
7. Decrement the length
8. Return the removed node.
##### Reverse:
1. Create a variable called current and set it to be the head of the list
2. Create a variable called tail and set it to be the head of the list
3. Loop through the list and set the next property of the current node to be the prev property of the current node
4. If there is no next property, set the tail to be the head and the head to be the current variable
5. Return the list
### Binary Search Tree:
#### Methods:
##### Insert:
1. Create a new node
2. Starting at the root
	* Check if there is a root, if not - the root now becomes that new node!
	* If there is a root, check if the value of the new node is greater than or less than the value of the root
	* If it is greater 
		* Check to see if there is a node to the right
			* If there is, move to that node and repeat these steps
			* If there is not, add that node as the right property
	* If it is less
		* Check to see if there is a node to the left
			* If there is, move to that node and repeat these steps
			* If there is not, add that node as the left property
##### Find:
1. Starting at the root
	1. Check if there is a root, if not - we're done searching!
	2. If there is a root, check if the value of the new node is the value we are looking for. If we found it, we're done!
	3. If not, check to see if the value is greater than or less than the value of the root
	4. If it is greater 
		1. Check to see if there is a node to the right
			1. If there is, move to that node and repeat these steps
			2. If there is not, we're done searching!
	5. If it is less
		1. Check to see if there is a node to the left
			1. If there is, move to that node and repeat these steps
			2. If there is not, we're done searching!
##### BFS (Breadth First Search):
1. Create a queue (this can be an array) and a variable to store the values of nodes visited
2. Place the root node in the queue
3. Loop as long as there is anything in the queue
	1. Dequeue a node from the queue and push the value of the node into the variable that stores the nodes
	2. If there is a left property on the node dequeued - add it to the queue
	3. If there is a right property on the node dequeued - add it to the queue
4. Return the variable that stores the values
##### DFS - In Order:
1. Create a variable to store the values of nodes visited
2. Store the root of the BST in a variable called current
3. Write a helper function which accepts a node
	1. If the node has a left property, call the helper function with the left property on the node
	2. Push the value of the node to the variable that stores the values
	3. If the node has a right property, call the helper function with the right property on the node
4. Invoke the helper function with the current variable
5. Return the array of values
##### DFS - Pre Order:
1. Create a variable to store the values of nodes visited
2. Store the root of the BST in a variable called current
3. Write a helper function which accepts a node
	1. Push the value of the node to the variable that stores the values
	2. If the node has a left property, call the helper function with the left property on the node
	3. If the node has a right property, call the helper function with the right property on the node
4. Invoke the helper function with the current variable
5. Return the array of values
##### DFS - Post Order:
1. Create a variable to store the values of nodes visited
2. Store the root of the BST in a variable called current
3. Write a helper function which accepts a node
	1. If the node has a left property, call the helper function with the left property on the node
	2. If the node has a right property, call the helper function with the right property on the node
	3. Push the value of the node to the variable that stores the values
4. Invoke the helper function with the current variable
5. Return the array of values
##### Remove:
1. Find the parent of the node that needs to be removed and the node that needs to be removed
2. If the value we are removing is greater than the parent node
3. Set the right property of the parent to be null
4. If the value we are removing is less than the parent node​
5. Set the left property of the parent to be null
6. Otherwise, the node we are removing has to be the root, so set the root to be null
##### Remove (1 Child):
1. Find the parent of the node that needs to be removed and the node that needs to be removed
2. See if the child of the node to be removed is on the right side or the left side
3. If the value we are removing is greater than the parent node​​
4. Set the right property of the parent to be the child
5. If the value we are removing is less than the parent node​
6. Set the left property of the parent to be the child
7. Otherwise, set the root property of the tree to be the child
##### Remove (2 Children):
1. Find the parent of the node that needs to be removed and the node that needs to be removed
2. Find the predecessor node and store that in a variable
3. Set the left property of the predecessor node to be the left property of the node that is being removed
4. If the value we are removing is greater than the parent node​​
	1. Set the right property of the parent to be the right property of the node to be removed
5. If the value we are removing is less than the parent node​
	1. Set the left property of the parent to be the right property of the node to be removed
6. Otherwise, set the root of the tree to be the right property of the node to be removed
