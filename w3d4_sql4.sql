## LAB 1 - GROUP BY ##

/* In the table actor, what last names are not repeated? 
For example if you would sort the data in the table actor by last_name, you would see that there is 
Christian Arkoyd, Kirsten Arkoyd, and Debbie Arkoyd. These three actors have the same last name. 
So we do not want to include this last name in our output. Last name "Astaire" is present only one time 
with actor "Angelina Astaire", hence we would want this in our output list.
*/
use sakila;

select distinct(last_name), count(last_name) as freq from actor
group by last_name 
having freq = 1;


/*
Which last names appear more than once? We would use the same logic as in the previous question 
but this time we want to include the last names of the actors where the last name was present more than once
*/
select last_name, count(last_name) as freq from actor
group by last_name 
having freq > 1;


-- Using the rental table, find out how many rentals were processed by each employee.
select count(rental_id) as num_rentals, concat(first_name, " ", last_name) as Employee from rental
join staff using(staff_id)
group by staff_id;


-- Using the film table, find out how many films there are of each rating.
select distinct(rating), count(title) as num_films from film
group by rating;


-- What is the mean length of the film for each rating type. 
-- Round off the average lengths to two decimal places
select distinct(rating), round(avg(length), 2) as "Average length" from film
group by rating;


-- Which kind of movies (rating) have a mean duration of more than two hours?
select distinct(rating), round(avg(length), 2) as avg_len from film
group by rating
having avg_len >= 120;



## LAB 2 