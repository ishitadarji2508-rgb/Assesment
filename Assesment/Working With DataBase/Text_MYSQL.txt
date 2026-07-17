# ASSESSMENT OF MYSQL
create table city (
id int primary key ,
city_name varchar(20),
lat decimal(11,7),
lon_g decimal(11,7),
country_id int
);
 create table customer (
 id int primary key,
 customer_name varchar(20),
 city_id int,
 customer_address varchar(20),
 next_call_date date,
 ts_inserted datetime(4),
 foreign key (id) references city(id)
 
 );
 use  query;
 create table country (
 id int primary key,
 country_name varchar(30),
 country_name_eng varchar(30),
 country_code varchar(30),
 foreign key (id) references city(id)
 );
 
SELECT 
    co.country_name_eng AS country_name,
    ci.city_name,
    cu.customer_name
FROM 
    country co
LEFT JOIN 
    city ci ON co.id = ci.country_id
LEFT JOIN 
    customer cu ON ci.id = cu.city_id;
 -- Retrieve country name, city name, and customer name using LEFT JOINs to include all countries, even those without cities or customers.

 SELECT 
    co.country_name_eng AS country_name,
    ci.city_name,
    cu.customer_name
FROM 
    country co
INNER JOIN 
    city ci ON co.id = ci.country_id
LEFT JOIN 
    customer cu ON ci.id = cu.city_id;
 -- Fetch countries with cities and optional customers
 
