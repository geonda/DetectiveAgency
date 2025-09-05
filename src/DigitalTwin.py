from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class City:
    def __init__(
        self, population: int = 1000, average_age: int = 20, std_age: int = 50
    ) -> None:
        self.population = population
        self.average_age = average_age
        self.std_age = std_age
        self.id = 0
        self.normal_dist = iter(
            np.random.normal(self.average_age, self.std_age, self.population)
        )

    def get_person_from_city(self):
        self.id += 1
        while True:
            age = next(self.normal_dist)
            if 14 < age < 90:
                break
        sex = np.random.choice(["male", "female"])
        problem_type = np.random.choice(["murder", "theft", "adultery"])
        return Person(self.id, age, sex, problem_type)

    def get_people(self, n):
        return [self.get_person_from_city() for _ in range(n)]

    def __repr__(self) -> str:
        return f"Created city with population of {self.population}"


class Person:
    def __init__(self, id, age, sex, problem_type):
        self.id = id
        self.age = age
        self.sex = sex
        self.problem_type = problem_type


class Detective:
    _id_count = 0

    def __init__(self, config: Dict[str, Dict[str, Any]], name: Optional[str] = None):
        self.id = Detective._id_count
        Detective._id_count += 1
        self.name = name
        self.ncases = {k: 0 for k in config.keys()}
        self.price = {k: item["price"] for k, item in config.items()}
        self.success_rates = {k: item["success_rate"] for k, item in config.items()}
        self.avg_time = {k: item["avg_time"] for k, item in config.items()}


class Case:
    _id_counter = 0

    def __init__(self, person: Person, detective: Detective):
        self.id = Case._id_counter
        Case._id_counter += 1
        self.person = person
        self.detective = detective
        self.problem_type = person.problem_type
        self.price = detective.price[person.problem_type]
        self.detective_id = detective.id

        self.local_id = detective.ncases[person.problem_type]
        self.status = "ongoing"
        self.time_init = self.init_date(datetime(2000, 1, 1), datetime(2025, 12, 31))
        self.time_finish = None
        self.time_spent = None

    def init_date(self, start_date, end_date):
        delta = end_date - start_date
        delta_seconds = delta.total_seconds()
        random_seconds = np.random.uniform(0, delta_seconds)
        return start_date + timedelta(seconds=random_seconds)

    def finish_date(self):
        mean_seconds = timedelta(
            days=self.detective.avg_time[self.problem_type]
        ).total_seconds()
        std_seconds = mean_seconds * 0.1
        finish = np.random.normal(mean_seconds, std_seconds)
        return self.time_init + timedelta(seconds=finish)


class Agency:
    def __init__(self, detectives: list[Detective]):
        self.detectives = detectives
        self.cases = None
        self.case_types = list(detectives[0].ncases.keys()) if detectives else []
        self.case_count = 0
        self.revenue = {detective.id: 0 for detective in detectives}
        self.distribution = {}

    def generate_distributions(self):
        for detective in self.detectives:
            self.distribution[detective.id] = {}
            for case_type in self.case_types:
                self.distribution[detective.id][case_type] = np.random.binomial(
                    n=1,
                    p=detective.success_rates[case_type],
                    size=detective.ncases[case_type],
                )

    def _process_case(self, case):
        case.time_finish = case.finish_date()
        case.time_spent = case.time_finish - case.time_init
        case.status = self.distribution[case.detective.id][case.problem_type][
            case.local_id
        ]
        case.revenue = (
            None if isinstance(case.status, str) else case.status * case.price
        )
        self.revenue[case.detective.id] += case.detective.price[case.problem_type]

    def process_cases(self):
        if not self.cases:
            return
        try:
            for case in self.cases:
                self._process_case(case)
        except Exception as err:
            print(err)

    def get_cases(self, people):
        self.clients=people
        self.cases = [self._assign_case(person) for person in people]
        self.generate_distributions()

    def _assign_case(self, person):
        detective_id = np.random.choice([0, 1])
        case = Case(person, self.detectives[detective_id])
        self.detectives[detective_id].ncases[person.problem_type] += 1
        self.case_count += 1
        return case

    def get_dataframe(self):
        self.df = pd.DataFrame(
            {
                # 'case id': [case.id for case in cases],
                "status": [case.status for case in self.cases],
                "init_date": [case.time_init for case in self.cases],
                "finish_date": [case.time_finish for case in self.cases],
                "time_spent": [case.time_spent for case in self.cases],
                "detective_id": [case.detective_id for case in self.cases],
                "revenue": [case.revenue for case in self.cases],
                "case_type": [case.problem_type for case in self.cases],
            },
            index=[case.id for case in self.cases],
        )
        return self.df


class QuickStart:
    def __init__(
        self,
        rm=dict(
            murder=dict(price=2000, avg_time=2, success_rate=0.5),
            theft=dict(price=500, avg_time=3, success_rate=0.7),
            adultery=dict(price=800, avg_time=1, success_rate=0.5),
        ),
        sh=dict(
            murder=dict(price=2000, avg_time=10, success_rate=0.5),
            theft=dict(price=500, avg_time=10, success_rate=0.5),
            adultery=dict(price=800, avg_time=10, success_rate=0.5),
        ),
        population=10000,
        avg_age=30,
    ) -> None:
        self.config_RM = rm

        self.config_SH = sh

        self.town = City(population=population, average_age=avg_age)

        self.clients = self.town.get_people(1000)
        self.agency = Agency(
            [
                Detective(config=self.config_SH, name="SH"),
                Detective(config=self.config_RM, name="RM"),
            ]
        )

    def go(self):
        self.agency.get_cases(self.clients)
        self.agency.process_cases()
        return self.agency
