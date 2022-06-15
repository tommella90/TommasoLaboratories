
### Lab | SQL Queries - Join Two Tables ###
use sakila;

-- 1 Which actor has appeared in the most films?
select count(actor_id) as freq, actor_id, film_id, first_name, last_name
from film_actor
inner join sakila.actor using(actor_id)
group by actor_id
order by freq desc
limit 1;

-- 2 Most active customer (the customer that has rented the most number of films)
select count(customer_id) as freq, first_name, last_name 
from customer
inner join rental using(customer_id)
group by customer_id
order by freq desc
limit 1;

-- 3 List number of films per category.
select category_id, name, count(name) as freq from category
inner join film_category using(category_id)
group by category_id
order by freq desc
limit 10;

-- 4 Display the first and last names, as well as the address, of each staff member.
select concat(first_name, " ", last_name) as Name, address from staff
inner join address using(address_id);

-- 5 Display the total amount rung up by each staff member in August of 2005.
select concat(first_name, " ", last_name) as Name, left(payment_date, 7) as date, sum(amount) as total
from staff 
inner join payment using(staff_id)
where left(payment_date, 7) = "2005-08"
group by Name;

-- 5 again 
select a.first_name, a.last_name, sum(b.amount) as total_sale
from staff a 
join payment b 
using(staff_id)
where year(payment_date) = 2005 and month(payment_date) = 08
group by staff_id;

-- 6 List each film and the number of actors who are listed for that film.
select film_id, title, count(actor_id) as freq from film
left join film_actor using(film_id)
group by film_id, title
order by freq desc
limit 10 ;

-- 7 Using the tables payment and customer and the JOIN command, 
-- list the total paid by each customer. List the customers alphabetically by last name.
select sum(amount) as 'Total paid' , concat(first_name, " ", last_name) as Name
from payment 
inner join customer using(customer_id)
group by customer_id
order by last_name;


-- optional. Which is the most rented film? The answer is Bucket Brotherhood 
-- This query might require using more than one join statement. Give it a try.
select title, film_id, store_id, count(inventory_id) as freq from film
inner join inventory using(film_id)
inner join rental using(inventory_id)
group by title
order by freq desc
limit 1;


## LAB 2 ##
-- 1 Write a query to display for each store its store ID, city, and country.
select store_id, city, country from store 
inner join address using(address_id)
inner join city using(city_id)
inner join country using(country_id);


-- 2 Write a query to display how much business, in dollars, each store brought in.
select store_id, sum(amount) as total from store
inner join staff using(store_id)
inner join payment using(staff_id)
group by store_id;


-- 3 What is the average running time(length) of films by category?
select name, avg(length) as Lenght from film
inner join film_category using(film_id)
inner join category using(category_id)
group by category_id;


-- 4 Which film categories are longest(length) (find Top 5)? 
-- (Hint: You can rely on question 3 output.)
select name, avg(length) as Lenght from film
inner join film_category using(film_id)
inner join category using(category_id)
group by category_id
order by Lenght desc
limit 5;


-- 5 Display the top 5 most frequently(number of times) rented movies in descending order.
select count(rental_id) as freq, title from film
inner join inventory using(film_id)
inner join rental using(inventory_id)
group by title
order by freq desc
limit 5;


-- 6 List the top five genres in gross revenue in descending order.
