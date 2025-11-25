# Financial Fraud Detection - Dataset Hikayesi

# Hikaye
Bu dataset, mobil para transferi işlemlerini simüle eder ve normal işlemlerin yanı sıra dolandırıcılık faaliyetlerini de içerir. Amacı, finansal hizmetler sektöründeki gizlilik endişeleri nedeniyle kamuya açık veri setlerinin azlığı sorununu gidermektir. PaySim adlı bir simülatör kullanılarak oluşturulmuş bu veri seti, dolandırıcılık tespiti araştırmaları için gerçekçi bir ortam sunar. Dolandırıcılar, müşteri hesaplarının kontrolünü ele geçirerek fonları başka hesaplara transfer etmeyi ve ardından nakit olarak çekmeyi hedeflerler.

# İş Problemi
**İşlem Türüne Göre Dolandırıcılık**: Farklı işlem türlerinin (TRANSFER, CASH-OUT vb.) dolandırıcılık riskleri farklıdır.
**Dengesiz Veri**: Gerçek veri setlerinde olduğu gibi, dolandırıcılık işlemleri oldukça azdır.
**Bakiye Kontrolü**: Dolandırıclık işlemlerinin çoğu, gönderici ve alıcı hesaplarındaki bakiye değişimleri ile ilişkilidir.

# Dataset Özellikleri
**Toplam İşlem**: Yaklaşık 6.3 milyon işlem
**Zaman Aralığı**: 744 adım, yani 30 günlük bir süre.
**Özellikler**:
 * `step`: Simülasyon zaman birimi (saat)
 * `type`: İşlem türü (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
 * `amount`: İşlem tutarı
 * `nameOrig`: İşlemi başlatan müşteri
 * `oldbalanceOrg`: İşlem öncesi gönderici bakiyesi
 * `newbalanceOrig`: İşlem sonrası gönderici bakiyesi
 * `nameDest`: İşlemin alıcısı
 * `oldbalanceDest`: İşlem öncesi alıcı bakiyesi
 * `newbalanceDest`: İşlem sonrası alıcı bakiyesi
 * `isFraud`: Hedef değişken (1=Fraud, 0=Normal)
 * `isFlaggedFraud`: Belirli bir eşiği aşan yasa dışı tranferleri işaretler (200.000'den fazla tutar)

 Bu veri seti, dolandırıcılık tespiti için sadece işlem tutarı veya zaman gibi basit özellikler yerine, bakiye gibi daha karmaşık desenleri öğrenmeye olanak tanır.
 

