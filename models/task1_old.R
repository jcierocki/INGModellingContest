# Title     : TODO
# Objective : TODO
# Created by: jcierocki
# Created on: 27.04.2021

require(tidyverse)
require(forecast)
require(tseries)
require(lmtest)
require(magrittr)
require(lubridate)
require(tsibble)

rm(list = ls())

df <- readxl::read_xlsx("data/Lions_Den_data.xlsx") %>%
  set_colnames(c("date", "value")) %>%
  mutate(date = as_date(date)) %>%
  as_tsibble()

df %>% ggplot(aes(x = date, y = value)) +
  geom_line() +
  labs(y = NULL, title = "Coal consumption per capita") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme(plot.title = element_text(hjust = 0.5))

df %>%
  filter(year(date) < 1998L) %>%
  ggplot(aes(x = date, y = value)) +
  geom_line() +
  labs(y = NULL, title = "Coal consumption per capita before 1998") +
  scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m", limits = c(ymd("1993-12-01"), ymd("1997-12-01"))) +
  theme(plot.title = element_text(hjust = 0.5))

df %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 20) +
  scale_x_continuous(breaks = seq(0, 0.5, by = 0.05))

df %>%
  filter(year(date) >= 2014L & year(date) <= 2017L) %>%
  ggplot(aes(x = date, y = value)) +
  geom_line() +
  scale_x_date(date_breaks = "3 month", date_labels = "%Y-%m")

df$value %>% as.ts()

df$value %>% kpss.test(null = "Level")
df$value %>% adf.test()
df$value %>% pp.test()

df$value %>% Acf()
df$value %>% Pacf()

bc_lambda <- BoxCox.lambda(df$value + 1e5, method = "loglik")
# bc_lambda <- BoxCox.lambda(df$value + 1e5, method = "guerrero")

sarma1 <- df %>%
  # filter(year(date) < 2017L) %>%
  pull(value) %>%
  as.ts() %>%
  # BoxCox(bc_lambda) %>%
  Arima(
    order = c(0L,0L,0L),
    seasonal = list(order = c(0L, 0L, 7L)),
    optim.method = "Nelder-Mead"
  )

df <- df %>% mutate(fitted = fitted(sarma1))

summary(sarma1)
residuals(sarma1) %>% plot()
coeftest(sarma1)

df %>%
  filter(year(date) < 2017L) %>%
  pull(value) %>%
  as.ts() %>%
  BoxCox(bc_lambda) %>%
  plot(type = "l")

df %>%
  pivot_longer(cols = c(value, fitted)) %>%
  ggplot(aes(x = date, y = value, color = name)) +
  geom_line() +
  labs(y = NULL, title = "Coal consumption per capita") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme(plot.title = element_text(hjust = 0.5))

# df %>% write_csv("data/data_with_sarma_fits.csv")

ets_model <- df %>%
  filter(year(date) < 2019L) %>%
  pull(value) %>%
  as.ts() %>%
  ets(model = "AAA")

summary(ets_model)
forecast(ets_model, h = 11) %>% plot()

?ets