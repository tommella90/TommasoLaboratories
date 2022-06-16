-- get customers for which the borrowed amount is highet than the mean amount of all customers 
use sakila; 


-- 1 How many copies of the film Hunchback Impossible exist in the inventory system?
select count(film_id) as total from inventory
where film_id in (select film_id from film
	where title = 'Hunchback Impossible')
    group by film_id;


-- 2 List all films whose length is longer than the average of all the films.
select title, length from film 
having length > (select avg(length) from film);

set @avg_len := (select avg(length) as Lenght from film);
select title, length from film where length > @avg_len;


-- 3 Use subqueries to display all actors who appear in the film Alone Trip.
select first_name, last_name from actor
	where actor_id in (select actor_id from film
		inner join film_actor using(film_id)
		where title = 'Alone Trip');


-- 4 Sales have been lagging among young families, and you wish to target all family movies for a promotion. 
-- Identify all movies categorized as family films.
set @cat := (select category_id from category
	where name = 'Family');

select title from film_category
	inner join category using(category_id)
		inner join film using(film_id)
		where category_id = @cat ;


-- 5 Get name and email from customers from Canada using subqueries. 
-- Do the same with joins. Note that to create a join, you will have to identify the correct tables 
-- with their primary keys and foreign keys, that will help you get the relevant information.
set @canada := (select country from country
	where country = 'canada');

select email from customer
	inner join address using(address_id)
		inner join city using(city_id)
			inner join country using(country_id)
			where country = @canada;


/* 6
Which are films starred by the most prolific actor? 
Most prolific actor is defined as the actor that has acted in the most number of films. 
First you will have to find the most prolific actor and then use that actor_id to find 
the different films that he/she starred.
*/
set @prolifit_actor:= (select actor_id from film_actor
group by actor_id order by count(*) desc limit 1);

select distinct title, actor_id from film
	inner join film_actor using(film_id)
	where actor_id = @prolifit_actor;



/* 7 Films rented by most profitable customer. 
You can use the customer table and payment table to find the most profitable customer ie the customer 
that has made the largest sum of payments
*/
set @prof_cust := (select customer_id from payment
group by customer_id order by sum(amount) desc limit 1);

select distinct title from film 
	inner join inventory using(film_id)
		inner join rental using(inventory_id)
		where customer_id = @prof_cust;

## other way
with
most_profilic_actor AS (select actor_id from film_actor
		group by actor_id order by count(film_id) desc limit 1)
select Title
from film inner join film_actor using (film_id)
where actor_id = (select actor_id from most_profilic_actor); 


-- 8 Customers who spent more than the average payments.
replace table1 (select sum(amount) as groupAmount from payment group by customer_id );

select distinct first_name, last_name, sum(amount) as total from customer
	inner join payment using(customer_id)
	group by first_name, last_name
		having total > (select avg(groupAmount) from table1)
		order by first_name;


## TEMPORARY TABLE 
create temporary table temp_table (select sum(amount) as groupAmount from payment group by customer_id );

select distinct first_name, last_name, sum(amount) as total from customer
	inner join payment using(customer_id)
	group by first_name, last_name
		having total > (select avg(groupAmount) from temp_table)
		order by first_name;

## CREATE TABLE 
create table 'contacts' (
	'first_name' varchart(50) not null,
    'second_name' varchart(50) not null,
    'email' varchart(255) not null
    #engine=InnoDB default charset=utf8mb4 
    )
insert into 'lab_db', 'contacts'
('first_name', 
 'last_name', 
 'email')
 values
 ('Tommas', 
 'Ramella', 
 'tommaso.ramella90@',
 );