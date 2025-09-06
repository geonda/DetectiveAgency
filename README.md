# DetectiveAgency LuAndHonesty
Mock project for A/B testing

Here is the deal..  There are two dectives in the agency and there is an advertising compain going on. Your mission, shell you choose accept it, is:
- Check who is the best detective in the agency. (done)
<!-- - Find out if the advertising launched on MAX is good idea or bad. -->
- Does green or redis the best color for the button on the website. (done)
- Create a set of metrics for the agency that should reflect both perfomance, revenue and potential growth? ()

The people how are seekig the help in the agency have following metadata:

    age 14-90
    sex
    problem_type=['private eye', 'murder', 'theft']
    time_init  DD-MM-YYYY-hh-ss
    time_finish DD-MM-YYYY-hh-ss
    status= success, failed, ongoing

Frist Detective RM:

    good on murder cases
    takes on average 2 days, 10% failure
    bad on missions with adultery
    takes 1 day  80% failure
    ok on theft cases 
    takes 3 days 60% failure
    if it a woman: 
        +10% increase in kpi

Second Decetive SH:

    ok on murder cases
    takes on average 1 days, 40% failure
    good on missions with adultery
    takes 2 day  20% failure
    ok on theft cases 
    takes 3 days 50% failure
    if the age is >50:
        - 10% in kpis


Website:

    Single button - call the agency

Prices:
    
    Murder case - 2000
    Theft case - 500
    Private eye - 800