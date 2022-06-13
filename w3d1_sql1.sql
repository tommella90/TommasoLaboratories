-- LAB 1. SAKILA
-- Explore tables by selecting all columns from each table or using the in built review features for your client
use sakila;
SELECT * FROM actor;
SELECT * FROM address;

-- Select one column from a table. Get film titles.
SELECT title FROM sakila.film title;

/*
Select one column from a table and alias it. Get unique list of film languages under the alias language. 
Note that we are not asking you to obtain the language per each film,
 but this is a good time to think about how you might get that information in the future
 */
 
use sakila;
SELECT * from film;
SELECT original_language_id as "Language" FROM sakila.film title;

-- Find out how many stores does the company have?
SELECT count(store_id) from store;

-- Find out how many employees staff does the company have?
SELECT * from staff;
SELECT count(staff_id) from staff;

-- Return a list of employee first names only?
SELECT first_name from staff;



-- LAB 2. BANKS
-- Get the id values of the first 5 clients from district_id with a value equals to 1.
use bank;
select client_id, district_id from client 
where district_id =1 order by client_id 
limit 5


-- In the client table, get an id value of the last client where the district_id equals to 72.
select client_id, district_id from bank.client
where district_id = 72 
order by client_id desc
limit 1


-- Get the 3 lowest amounts in the loan table.
select amount from bank.loan
order by amount
limit 3


-- What are the possible values for status, ordered alphabetically in ascending order in the loan table?
select distinct status from bank.loan 
order by status 


-- What is the loan_id of the highest payment received in the loan table?
select payments, loan_id status from bank.loan 
order by payments desc
limit 1 

-- What is the loan amount of the lowest 5 account_ids in the loan table? Show the account_id and the corresponding amount
select amount, account_id status from bank.loan 
order by account_id
limit 5


-- What are the top 5 account_ids with the lowest loan amount that have a loan duration of 60 in the loan table?
select amount, account_id, duration from bank.loan 
where duration = 60 
order by amount
limit 5


-- What are the unique values of k_symbol in the order table?
select distinct k_symbol from bank.order
where k_symbol is not null


-- In the order table, what are the order_ids of the client with the account_id 34?
select distinct order_id from bank.order
where account_id = 34


-- In the order table, which account_ids were responsible for orders between order_id 29540 and order_id 29560 (inclusive)?
select distinct account_id from bank.order
where order_id between 29540 and 29560


-- In the order table, what are the individual amounts that were sent to (account_to) id 30067122?
select distinct amount from bank.order
where account_to = 30067122


-- In the trans table, show the trans_id, date, type and amount
-- of the 10 first transactions from account_id 793 in chronological order, from newest to oldest.
select trans_id, date, type, amount, account_id from bank.trans
where account_id = 793
order by trans_id, account_id 
limit 10


-- In the client table, of all districts with a district_id lower than 10, 
-- how many clients are from each district_id? Show the results sorted by the district_id in ascending order.
SELECT count(client_id), district_id from bank.client
where district_id <= 10
group by district_id
order by district_id


-- In the card table, how many cards exist for each type? Rank the result starting with the most frequent type.
select type, count(card_id) from bank.card
group by(type) 
order by card_id desc


-- Using the loan table, print the top 10 account_ids based on the sum of all of their loan amounts.
select account_id, amount from bank.loan
order by amount desc
limit 10


-- In the loan table, retrieve the number of loans issued for each day, 
-- before (excl) 930907, ordered by date in descending order.
select count(distinct(loan_id)), date from bank.loan
where date < 930907
group by date 
order by date desc


-- In the loan table, for each day in December 1997, count the number of loans issued for each unique loan duration, 
-- ordered by date and duration, both in ascending order. You can ignore days without any loans in your output.
select count(loan_id), date, duration from bank.loan
where date between 971200  and 971230
group by date, duration 
order by date, duration


-- In the trans table, for account_id 396, sum the amount of transactions for each type 
-- (VYDAJ = Outgoing, PRIJEM = Incoming). Your output should have the account_id, the type and the sum of amount, 
-- named as total_amount. Sort alphabetically by type.
select account_id, type, sum(amount) from bank.trans
where account_id = 396
group by type


-- From the previous output, translate the values for type to English, 
-- rename the column to transaction_type, round total_amount down to an integer
select account_id, round(sum(amount)) as "Amount",
CASE 
   WHEN type LIKE 'VYDAJ' THEN 'Outgoing'   
   WHEN type LIKE 'PRIJEM' THEN 'Incoming'   
end as type
from bank.trans
where account_id = 396
group by type;


-- From the previous result, modify your query so that it returns only one row,
-- with a column for incoming amount, outgoing amount and the difference.
WITH group1 AS (
select account_id, type, round(sum(amount)) as "Amount1"
from bank.trans
where account_id = 396 and type = "VYDAJ"
group by type ),
group2 AS (
select account_id, type, round(sum(amount)) as "Amount2"
from bank.trans
where account_id = 396 and type = "PRIJEM"
group by type)
SELECT *
	FROM group1 JOIN group2 USING (account_id);


-- From the previous result, modify your query so that it returns only one row,
-- with a column for incoming amount, outgoing amount and the difference.
WITH group1 AS (
select account_id, type as type1, round(sum(amount)) as "Amount1"
from bank.trans
where account_id = 396 and type = "VYDAJ"
group by type ),
group2 AS (
select account_id, type as type2, round(sum(amount)) as "Amount2"
from bank.trans
where account_id = 396 and type = "PRIJEM"
group by type), 
group3 as (
SELECT *
	FROM group1 JOIN group2 USING (account_id))
select account_id, group3.Amount1 as "Outgoing", group3.Amount2 as "Incoming", group3.Amount2 - group3.Amount1 as "difference"
from group3;


-- Continuing with the previous example, rank the top 10 account_ids based on their difference.
with gr1 as (
select account_id, round(sum(amount)) as "Outgoing" from bank.trans
where type = "PRIJEM" 
group by account_id),
gr2 as (
select account_id, round(sum(amount)) as "Incoming" from bank.trans
where type = "VYDAJ" 
group by account_id),
gr3 as (
select *
	FROM gr1 JOIN gr2 USING (account_id))
select account_id, gr3.Outgoing, gr3.Incoming, gr3.Outgoing - gr3.Incoming as difference
from gr3
order by difference desc
limit 10
