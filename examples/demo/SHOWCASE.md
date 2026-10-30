# Показ: заявки службы поддержки

Документы содержат только синтетические данные. Запустите API и worker по README. Для двух основных кадров достаточно native PDF/table режима и Poppler. Третий кадр требует настроенных OCR, Tesseract и OpenCV до загрузки схемы.

## Подготовка

Загрузите `support-handbook.txt`, `request-register.pdf` и, если включён OCR/OpenCV, `approval-flow.pdf`. Дождитесь ready. Выдайте viewer право READ через API администратора и подключитесь viewer-токеном.

```text
POST /v1/admin/documents/<document-id>/permissions
Authorization: Bearer <admin-token>
Content-Type: application/json

{"principal_type":"user","principal_id":"<viewer-user-id>","permission":"read"}
```

Идентификатор viewer находится в RAG_DEMO_VIEWER_USER_ID локального `.env`. Реальные токены не должны попадать в скриншоты или Git.

## Общий вид приложения

Выберите Options → Page answer · local. Поле Find source оставьте пустым. Введите `Which Priority is listed for VPN access?`. Ожидается `High` и ссылка Request register. Сверните Options, чтобы были видны список документов, вопрос, ответ и источник.

## Таблица с источником

Нажмите View on page. Видна строка VPN access и ячейка High. Другой пример: `Which Status is listed for VPN access?` → `Approved`.

## Ветка согласования

В Options задайте Find source: `Request approval`. Введите `Which step follows Review when Yes?` → `Approve`. Откройте View on page. Для ветки No ожидается `Clarify`. Без условия (`Which step follows Review?`) ожидается отказ от ответа. Фильтр источника нужен в этом сценарии; без него подходящая страница может не найтись.

Перед возвращением к вопросам по таблице очистите Find source. Не делайте несколько одинаковых копий реестра под разными ключами: это может создать корректную неоднозначность между документами.

Эти сценарии проверены на настоящем локальном API без вызова языковой модели. Примеры не являются независимой оценкой качества. `approval-flow.png` — исходное изображение схемы для просмотра; в приложение загружается PDF.
