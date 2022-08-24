using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class guardWeapon : MonoBehaviour
{
    public Transform firePoint;
    public GameObject guardObj;
    private GameObject playerObj = null;
    public GameObject bulletPrefab;
    public AudioSource audioPlayer_fireball;
    


    public float shootRadius;
    private float timeBtwShots;
    public float startTimeBtwShots;

    private void Start()
    {
        if (playerObj == null)
        {
            playerObj = GameObject.FindGameObjectWithTag("Player");
        }

        timeBtwShots = startTimeBtwShots;

    }

    // Update is called once per frame
    void Update()
    {
        Vector2 playerLoc = playerObj.transform.position;
        Vector2 guardLoc = guardObj.transform.position;
        if (Vector2.Distance(playerLoc, guardLoc) <= shootRadius){
            if (timeBtwShots <= 0)
            {
                Shoot();
                timeBtwShots = startTimeBtwShots;
            }
            else
            {
                timeBtwShots -= Time.deltaTime;
            }

        }

    }

    void Shoot()
    {
        // Shooting Logic
        Instantiate(bulletPrefab, firePoint.position, Quaternion.identity);
        audioPlayer_fireball.Play();
    }



}
